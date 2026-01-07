"""
End-to-End (E2E) Agent for IALM project.

This agent combines the functionality of DST, DP, and NLG agents into a single end-to-end agent.
"""

import json
import logging
import re
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from .base_agent import BaseAgent
from ..core.ssm import SharedStructuredMemory
from ..esb.esb_evolver import ESBEvolver as HSMEvolver
from ..models.belief_state import parse_belief_state_from_response

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.multiwoz21.database import Database as MultiWOZDatabase
from data.sgd.database import Database as SGDDatabase

logger = logging.getLogger(__name__)


class E2EAgent(BaseAgent):
    """
    End-to-End (E2E) Agent for task-oriented dialogue systems.
    
    This agent combines the functionality of DST, DP, and NLG agents into a single end-to-end agent.
    It handles dialogue state tracking, dialogue policy, and natural language generation in one step.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None, dataset_type: str = "multiwoz"):
        """
        Initialize the E2E Agent.
        
        Args:
            config: Agent configuration dictionary
            llm_client: LLM client instance
            ssm: SharedStructuredMemory instance
            hsm_evolver: HSMEvolver instance for strategy retrieval
            dataset_type: Dataset type, either "multiwoz" or "sgd"
        """
        super().__init__(config, llm_client, ssm, hsm_evolver)
        self.agent_type = "e2e"
        self.dataset_type = dataset_type
        
        # Initialize database based on dataset type
        self.db = None
        if self.dataset_type == "multiwoz" and MultiWOZDatabase:
            self.db = MultiWOZDatabase()
        elif self.dataset_type == "sgd" and SGDDatabase:
            self.db = SGDDatabase()

    def process(self, domains: List[str], user_utterance: str, dialogue_history: List[str], pre_belief_state: Dict[str, Any], hsm: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the input data and generate the complete dialogue response.
        
        Args:
            domains: Dialogue domains
            user_utterance: User utterance
            dialogue_history: Dialogue history
            pre_belief_state: Previous belief state
            hsm: HSM strategies
            
        Returns:
            Dictionary containing:
            - belief_state: Updated belief state
            - reason: Reason for the output
            - criticism: Criticism of the input (if any)
            - system_action: Generated system action
            - db_query_needed: Whether a database query was needed
            - query: Database query parameters (if any)
            - db_results: Database query results (if any)
            - system_utterance: Generated system utterance
        """
        
        # First call to LLM to get belief state, system action, and determine if DB query is needed
        prompt = self._construct_prompt(domains, user_utterance, dialogue_history, pre_belief_state, hsm)
        
        try:
            messages = [
                {"role": "system", "content": "You are an End-to-End Agent for a task-oriented dialogue system. Combine dialogue state tracking, dialogue policy, and natural language generation. Always query database for entity information, never use your own knowledge."},
                {"role": "user", "content": prompt}
            ]
            
            # Call LLM
            response = self.llm_client.generate_with_chat_format(messages=messages)
            cleaned_response = self.llm_client.clean_response(response)
            parsed_response = json.loads(cleaned_response)
            
            # Extract results from first LLM call
            belief_state = parse_belief_state_from_response(pre_belief_state, parsed_response.get("belief_state", {}))
            reason = parsed_response.get("reason", "")
            criticism = parsed_response.get("criticism", "")
            system_action_raw = parsed_response.get("system_action", "")
            db_query_needed = parsed_response.get("db_query_needed", False)
            query = parsed_response.get("query", {})
            
            # Ensure system_action is a string
            if isinstance(system_action_raw, str):
                system_action = system_action_raw
            else:
                logger.warning(f"system_action is not a string: {type(system_action_raw)}, value: {system_action_raw}")
                system_action = str(system_action_raw)
            
            db_results = []
            system_utterance = ""
            
            if db_query_needed:
                # Execute database query using the same method as DPAgent
                db_results = self._execute_database_query(query)
                
                # Second call to LLM to generate final system utterance with DB results
                prompt_with_db = self._construct_prompt_with_db_results(
                    domains, user_utterance, dialogue_history, belief_state, 
                    system_action, db_results, hsm
                )
                
                messages = [
                    {"role": "system", "content": "You are an End-to-End Agent for a task-oriented dialogue system. Generate natural language responses based on belief state, system action, and database results."},
                    {"role": "user", "content": prompt_with_db}
                ]
                
                response = self.llm_client.generate_with_chat_format(messages=messages)
                cleaned_response = self.llm_client.clean_response(response)
                parsed_response = json.loads(cleaned_response)
                
                system_utterance = parsed_response.get("system_utterance", "")
            else:
                # No DB query needed, generate system utterance directly
                system_utterance = parsed_response.get("system_utterance", "")
            
            return {
                "belief_state": belief_state,
                "reason": reason,
                "criticism": criticism,
                "system_action": system_action,
                "db_query_needed": db_query_needed,
                "query": query,
                "db_results": db_results,
                "system_utterance": system_utterance
            }
            
        except Exception as e:
            logger.error(f"Error in E2EAgent.process: {str(e)}")
            return {
                "belief_state": pre_belief_state,
                "reason": f"Error occurred: {str(e)}",
                "criticism": "",
                "system_action": "request(general->clarification)",
                "db_query_needed": False,
                "query": {},
                "db_results": [],
                "system_utterance": "I didn't understand, could you please say that again."
            }
            
    def _construct_prompt(self, domains: List[str], user_utterance: str, dialogue_history: List[str], 
                         pre_belief_state: Dict[str, Any], hsm: List[Dict[str, Any]]) -> str:
        """
        Construct prompt for the first LLM call (without DB results).
        
        Args:
            domains: Dialogue domains
            user_utterance: User utterance
            dialogue_history: Dialogue history
            pre_belief_state: Previous belief state
            hsm: HSM strategies
            
        Returns:
            Constructed prompt string
        """
        # Format dialogue history
        formatted_history = ""
        if len(dialogue_history) > 0:
            formatted_history = "## Dialogue History:" + "\n".join(dialogue_history)
        
        # Format HSM strategies
        formatted_hsm = ""
        if hsm and len(hsm) > 0:
            formatted_hsm = "## Relevant Strategies:\n"
            for strategy in hsm:
                if isinstance(strategy, dict):
                    formatted_hsm += strategy.get('content', '') + "\n"
        
        # Construct the full prompt
        prompt = f"""
## END-TO-END AGENT
- Domain(s): {', '.join(domains)}
- User Utterance: {user_utterance}
- Previous Belief State: {json.dumps(pre_belief_state, indent=2)}

{formatted_history}

{formatted_hsm}

## INSTRUCTIONS:
1. First, analyze the user utterance and update the belief state. Only modify existing slots, never add new slots or domains.
2. If there are issues with the user output, provide criticism. If good, leave criticism as empty string.
3. Determine the appropriate system action based on the updated belief state.
4. Decide if a database query is needed (set db_query_needed to true/false).
5. If db_query_needed is true, specify the query parameters using the filled slots.
6. If db_query_needed is false, generate the system utterance directly.
7. Provide a reason for your output.

## SYSTEM ACTION TYPES:
- inform(slot=value): Provide information to the user about a specific slot
- request(slot): Request more information from the user for a specific slot
- recommend(entity): Recommend a specific entity to the user
- select(entity): Select an entity from the database results
- nooffer(): Inform user that no matching results were found
- book(slot1=value1,slot2=value2): Make a booking with specified parameters
- nobook(): Inform user that booking cannot be completed
- offerbook(slot1=value1,slot2=value2): Offer booking options with specified parameters
- offerbooked(booking_details): Confirm successful booking with details

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "criticism": "Your criticism of the User Output (if any)",
  "belief_state": {{"domain1": {{"slot1": "value1", "slot2": "value2"}}, "domain2": {{...}}}},
  "system_action": "appropriate action based on the current context",
  "reason": "Your reason for the output",
  "db_query_needed": true/false,
  "query": {{
    "domain": "the name of domain to query",
    "state": {{"the name of domain to query": {{"slot_name1": "value1", "slot_name2": "value2"}} }}
  }},
  "system_utterance": "your natural system response (only if db_query_needed is false)"
}}
"""
        
        return prompt
    
    def _construct_prompt_with_db_results(self, domains: List[str], user_utterance: str, dialogue_history: List[str], 
                                         belief_state: Dict[str, Any], system_action: str, db_results: List[Dict[str, Any]], 
                                         hsm: List[Dict[str, Any]]) -> str:
        """
        Construct prompt for the second LLM call (with DB results).
        
        Args:
            domains: Dialogue domains
            user_utterance: User utterance
            dialogue_history: Dialogue history
            belief_state: Updated belief state
            system_action: System action
            db_results: Database query results
            hsm: HSM strategies
            
        Returns:
            Constructed prompt string
        """
        # Format dialogue history
        formatted_history = ""
        if len(dialogue_history) > 0:
            formatted_history = "## Dialogue History:" + "\n".join(dialogue_history)
        
        # Format HSM strategies
        formatted_hsm = ""
        if hsm and len(hsm) > 0:
            formatted_hsm = "## Relevant Strategies:\n"
            for strategy in hsm:
                if isinstance(strategy, dict):
                    formatted_hsm += strategy.get('content', '') + "\n"
        
        # Format database results
        formatted_db_results = ""
        if len(db_results) > 0:
            formatted_db_results = "## Database Results: " + json.dumps(db_results)
        
        # Construct the full prompt
        prompt = f"""
## END-TO-END AGENT (with Database Results)
- Domain(s): {', '.join(domains)}
- User Utterance: {user_utterance}
- Current Belief State: {json.dumps(belief_state, indent=2)}
- System Action: {system_action}
{formatted_db_results}

{formatted_history}

{formatted_hsm}

## INSTRUCTIONS:
1. Analyze the database results and system action.
2. Generate a natural language system utterance based on the system action and database results.
3. Ensure the response is natural, helpful, and appropriate for the dialogue context.
4. Keep the response concise but informative.
5. Provide a reason for your output.

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "system_utterance": "your natural system response",
  "reason": "Your reason for the output"
}}
"""
        
        return prompt
    
    def _execute_database_query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute database query using the same method as DPAgent.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of database results
        """
        if not self.db:
            logger.warning("Database not initialized")
            return []
            
        try:                
            # Check domain parameter
            query_domain = query_params.get("domain", None)
            if not query_domain:
                logger.error("Domain not specified in query_params")
                return []
                
            # Get state parameter
            state = query_params.get("state", {})
            if not state:
                logger.warning("State not specified in query_params, using empty state")
                state = {}
                
            # Get other optional parameters with default values
            topk = query_params.get("topk", 5)
            ignore_open = query_params.get("ignore_open", False)
            soft_constraints = query_params.get("soft_constraints", [])
            fuzzy_match_ratio = query_params.get("fuzzy_match_ratio", 60)
            
            # Record query parameters
            logger.debug(f"Executing database query with params: domain={query_domain}, state={state}, topk={topk}")
            
            # Execute query
            results = self.db.query(
                domain=query_domain,
                state=state,
                topk=topk,
                ignore_open=ignore_open,
                soft_contraints=soft_constraints,  # Note: database function uses soft_contraints (without 's')
                fuzzy_match_ratio=fuzzy_match_ratio
            )
            
            # Verify results
            if not isinstance(results, list):
                logger.error(f"Invalid query result type: {type(results)}, expected list")
                return []
                
            logger.info(f"Database query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            return []
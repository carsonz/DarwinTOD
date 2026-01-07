"""
LLM client utility for communicating with various LLM APIs compatible with OpenAI interface.
"""

import json
import re
import time
import random
import yaml
import os
import logging
from typing import Dict, Any, List
from openai import OpenAI


class LLMClient:
    """Client for communicating with various LLM APIs compatible with OpenAI interface."""
    
    def __init__(self, provider: str = "openai", config_path: str = "config/default_config.yaml", 
                 base_url: str = None, model: str = None, api_key: str = None, timeout: float = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use (openai, qwen, deepseek, minimax, vllm)
            config_path: Path to configuration file
            base_url: Base URL for the API (optional, will use config if not provided)
            model: Model name to use (optional, will use config if not provided)
            api_key: API key to use (optional, will use config if not provided)
            timeout: Timeout for API requests in seconds (optional, will use config if not provided)
        """
        # 禁用httpx的INFO级别日志，减少HTTP请求日志输出
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Load configuration
        self.provider = provider
        self.config_path = config_path
        self.config = self._load_config()
        
        # Get provider configuration
        provider_config = self.config.get("api", {}).get(provider, {})
        
        # Use provided parameters or fall back to config
        self.base_url = (base_url or provider_config.get("base_url", "https://api.openai.com/v1")).rstrip('/')
        self.model = model or provider_config.get("model", "gpt-3.5-turbo")
        self.api_key = api_key or provider_config.get("api_key", "")
        self.max_retries = provider_config.get("max_retries", 3)
        self.retry_delay = provider_config.get("retry_delay", 0.5)
        self.max_retry_delay = provider_config.get("max_retry_delay", 10)
        self.timeout = timeout or provider_config.get("timeout", 180)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # If config file doesn't exist, return empty dict
        if not os.path.exists(self.config_path):
            return {}
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for the API call."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5000, timeout: float = None) -> str:
        """
        Generate text using the API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout for this request in seconds (optional, overrides client default)
            
        Returns:
            Generated text
        """
        for attempt in range(self.max_retries):
            try:
                # Use completions API for all providers
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.timeout
                )
                # 检查响应是否有效
                if not response.choices or not response.choices[0]:
                    raise ValueError("Invalid response: empty choices")
                
                return response.choices[0].text.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"{self.provider} API call failed (attempt {attempt+1}): {str(e)}")
                # Calculate exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt)
                # Add jitter (random between 0.5x and 1.5x delay)
                delay *= random.uniform(0.5, 1.5)
                # Cap at max_retry_delay
                delay = min(delay, self.max_retry_delay)
                time.sleep(delay)
    
    def generate_with_chat_format(self, messages: List[Dict[str, str]], 
                                 temperature: float = 0.1, 
                                 max_tokens: int = 5000,
                                 extra_body: Dict[str, Any] = None,
                                 timeout: float = None) -> str:
        """
        Generate text using chat format messages.
        
        Args:
            messages: List of messages with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            extra_body: Additional parameters to pass to the API (e.g., {"reasoning_split": True})
            timeout: Timeout for this request in seconds (optional, overrides client default)
            
        Returns:
            Generated text
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare the completion call
                completion_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout or self.timeout
                }
                
                # Add extra_body if provided
                if extra_body:
                    completion_kwargs.update(extra_body)
                
                response = self.client.chat.completions.create(**completion_kwargs)
                
                # 检查响应是否有效
                if not response.choices or not response.choices[0] or not response.choices[0].message:
                    raise ValueError("Invalid response: empty choices or message")
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"{self.provider} API call failed (attempt {attempt+1}): {str(e)}")
                # Calculate exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt)
                # Add jitter (random between 0.5x and 1.5x delay)
                delay *= random.uniform(0.5, 1.5)
                # Cap at max_retry_delay
                delay = min(delay, self.max_retry_delay)
                time.sleep(delay)
    
    def generate_with_reasoning(self, messages: List[Dict[str, str]], 
                               temperature: float = 0.1, 
                               max_tokens: int = 500,
                               timeout: float = None) -> Dict[str, str]:
        """
        Generate text with reasoning details separated.
        
        Args:
            messages: List of messages with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout for this request in seconds (optional, overrides client default)
            
        Returns:
            Dictionary with 'reasoning' and 'content' keys
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"reasoning_split": True},
                    timeout=timeout or self.timeout
                )
                
                # 检查响应是否有效
                if not response.choices or not response.choices[0] or not response.choices[0].message:
                    raise ValueError("Invalid response: empty choices or message")
                
                reasoning = ""
                content = response.choices[0].message.content.strip()
                
                # Extract reasoning details if available
                if hasattr(response.choices[0].message, 'reasoning_details') and response.choices[0].message.reasoning_details:
                    reasoning = response.choices[0].message.reasoning_details[0]['text'].strip()
                
                return {
                    "reasoning": reasoning,
                    "content": content
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"{self.provider} API call failed (attempt {attempt+1}): {str(e)}")
                # Calculate exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt)
                # Add jitter (random between 0.5x and 1.5x delay)
                delay *= random.uniform(0.5, 1.5)
                # Cap at max_retry_delay
                delay = min(delay, self.max_retry_delay)
                time.sleep(delay)
    
    def clean_response(self, response: str) -> str:
        """
        Clean the model response by removing thinking process markers and other artifacts.
        Also extracts JSON from mixed text responses.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Cleaned response with thinking process removed and JSON extracted if present
        """
        # Remove any leading/trailing whitespace
        response = response.strip()

        # Remove thinking process markers like "</think>"
        if "</think>" in response:
            # Find all occurrences of the marker
            parts = response.split("</think>")
            if len(parts) > 1:
                # Take the content after the first marker
                response = parts[1].strip()

        # Extract JSON from the response
        # First, look for code blocks
        if "```json" in response:
            # Extract JSON from code block
            start_idx = response.find("```json") + 7
            end_idx = response.find("```", start_idx)
            if end_idx != -1:
                json_str = response[start_idx:end_idx].strip()
            else:
                # No closing code block, take everything after "```json"
                json_str = response[start_idx:].strip()
        elif "```" in response:
            # Extract from generic code block
            start_idx = response.find("```") + 3
            end_idx = response.find("```", start_idx)
            if end_idx != -1:
                json_str = response[start_idx:end_idx].strip()
            else:
                # No closing code block, take everything after "```"
                json_str = response[start_idx:].strip()
        else:
            # Look for JSON directly in the response
            # Find the first '{' or '['
            json_start = min(
                response.find('{') if response.find('{') != -1 else float('inf'),
                response.find('[') if response.find('[') != -1 else float('inf')
            )
            
            if json_start == float('inf'):
                # No JSON found, return empty string
                return "{}"
            
            # Find the last '}' or ']'
            json_end = max(
                response.rfind('}') if response.rfind('}') != -1 else -1,
                response.rfind(']') if response.rfind(']') != -1 else -1
            )
            
            if json_end == -1 or json_end <= json_start:
                # Invalid JSON, return empty string
                return "{}"
            
            json_str = response[json_start:json_end+1]
        
        # Clean the JSON string by removing any non-JSON content
        # Remove any leading text before the JSON
        if not (json_str.startswith('{') or json_str.startswith('[')):
            json_start = min(
                json_str.find('{') if json_str.find('{') != -1 else float('inf'),
                json_str.find('[') if json_str.find('[') != -1 else float('inf')
            )
            if json_start != float('inf'):
                json_str = json_str[json_start:]
        
        # Remove any trailing text after the JSON
        if not (json_str.endswith('}') or json_str.endswith(']')):
            json_end = max(
                json_str.rfind('}') if json_str.rfind('}') != -1 else -1,
                json_str.rfind(']') if json_str.rfind(']') != -1 else -1
            )
            if json_end != -1:
                json_str = json_str[:json_end+1]
        
        # Try to parse the JSON to verify it's valid
        try:
            # Try to parse the JSON to verify it's valid
            json.loads(json_str)
            # If parsing succeeds, return this JSON structure
            return json_str.strip()
        except json.JSONDecodeError:
            # If parsing fails, try to fix common issues
            # Remove any comments
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Try to fix unescaped quotes in JSON strings
            try:
                # A more robust approach to fix unescaped quotes in JSON
                # We'll use a state machine approach to handle quotes properly
                fixed_response = ""
                in_string = False
                escape_next = False
                
                for i, char in enumerate(json_str):
                    if escape_next:
                        fixed_response += char
                        escape_next = False
                    elif char == '\\':
                        fixed_response += char
                        escape_next = True
                    elif char == '"' and not in_string:
                        # This is a string delimiter
                        fixed_response += char
                        in_string = True
                    elif char == '"' and in_string:
                        # Check if this is an escaped quote or a string delimiter
                        # Look ahead to see if this is followed by a colon, comma, or closing brace
                        next_chars = json_str[i+1:i+3].strip()
                        if next_chars and next_chars[0] in [',', '}', ':']:
                            # This is likely a string delimiter
                            fixed_response += char
                            in_string = False
                        else:
                            # This is likely an unescaped quote within a string
                            fixed_response += '\\"'
                    else:
                        fixed_response += char
                
                json_str = fixed_response
                
                # Try to parse again
                json.loads(json_str)
                return json_str.strip()
            except:
                # If fixing fails, return an empty JSON object
                return "{}"
        
        # If all else fails, return an empty JSON object
        return "{}"
    

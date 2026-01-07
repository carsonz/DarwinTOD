def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={
                     'attraction': {'type': '', 'name': '', 'area': ''}, 
                     'hotel': {'name': '', 'area': '', 'parking': '', 'price range': '', 'stars': '', 'internet': 'yes', 'type': 'hotel', 'book stay': '', 'book day': '', 'book people': ''}, 
                     'restaurant': {'food': '', 'price range': '', 'name': '', 'area': '', 'book time': '', 'book day': '', 'book people': ''}, 
                     'taxi': {'leave at': '', 'destination': '', 'departure': '', 'arrive by': ''}, 
                     'train': {'leave at': '', 'destination': '', 'day': '', 'arrive by': '', 'departure': '', 'book people': ''}, 
                     'hospital': {'department': ''},
                     'police': {'name': '', 'address': '', 'phone': '', 'postcode': ''}
                     },
                 booked={},
                 request_state={},
                 terminated=False,
                 history=[])
    return state
def default_state():
    """
    Create a default state for SGD dialogue system based on the ontology structure.
    This function initializes the state with all domains and their respective slots.
    """
    state = {
        'user_action': [],
        'system_action': [],
        'belief_state': {
            # Banks domains
            'Banks_1': {
                'account_type': '',
                'recipient_account_type': '',
                'balance': '',
                'amount': '',
                'recipient_account_name': ''
            },
            'Banks_2': {
                'account_type': '',
                'recipient_account_type': '',
                'account_balance': '',
                'transfer_amount': '',
                'recipient_name': '',
                'transfer_time': ''
            },
            
            # Buses domains
            'Buses_1': {
                'from_location': '',
                'to_location': '',
                'from_station': '',
                'to_station': '',
                'leaving_date': '',
                'leaving_time': '',
                'fare': '',
                'travelers': '',
                'transfers': ''
            },
            'Buses_2': {
                'origin': '',
                'destination': '',
                'origin_station_name': '',
                'destination_station_name': '',
                'departure_date': '',
                'price': '',
                'departure_time': '',
                'group_size': '',
                'fare_type': ''
            },
            'Buses_3': {
                'from_city': '',
                'to_city': '',
                'from_station': '',
                'to_station': '',
                'departure_date': '',
                'departure_time': '',
                'price': '',
                'additional_luggage': '',
                'num_passengers': '',
                'category': ''
            },
            
            # Calendar domain
            'Calendar_1': {
                'event_date': '',
                'event_time': '',
                'event_location': '',
                'event_name': '',
                'available_start_time': '',
                'available_end_time': ''
            },
            
            # Events domains
            'Events_1': {
                'category': '',
                'subcategory': '',
                'event_name': '',
                'date': '',
                'time': '',
                'number_of_seats': '',
                'city_of_event': '',
                'event_location': '',
                'address_of_location': ''
            },
            'Events_2': {
                'event_type': '',
                'category': '',
                'event_name': '',
                'date': '',
                'time': '',
                'number_of_tickets': '',
                'city': '',
                'venue': '',
                'venue_address': ''
            },
            'Events_3': {
                'event_type': '',
                'event_name': '',
                'date': '',
                'time': '',
                'number_of_tickets': '',
                'price_per_ticket': '',
                'city': '',
                'venue': '',
                'venue_address': ''
            },
            
            # Flights domains
            'Flights_1': {
                'passengers': '',
                'seating_class': '',
                'origin_city': '',
                'destination_city': '',
                'origin_airport': '',
                'destination_airport': '',
                'departure_date': '',
                'return_date': '',
                'number_stops': '',
                'outbound_departure_time': '',
                'outbound_arrival_time': '',
                'inbound_arrival_time': '',
                'inbound_departure_time': '',
                'price': '',
                'refundable': '',
                'airlines': ''
            },
            'Flights_2': {
                'passengers': '',
                'seating_class': '',
                'origin': '',
                'destination': '',
                'origin_airport': '',
                'destination_airport': '',
                'departure_date': '',
                'return_date': '',
                'number_stops': '',
                'outbound_departure_time': '',
                'outbound_arrival_time': '',
                'inbound_arrival_time': '',
                'inbound_departure_time': '',
                'fare': '',
                'is_redeye': '',
                'airlines': ''
            },
            'Flights_3': {
                'passengers': '',
                'flight_class': '',
                'origin_city': '',
                'destination_city': '',
                'origin_airport_name': '',
                'destination_airport_name': '',
                'departure_date': '',
                'return_date': '',
                'number_stops': '',
                'outbound_departure_time': '',
                'outbound_arrival_time': '',
                'inbound_arrival_time': '',
                'inbound_departure_time': '',
                'price': '',
                'number_checked_bags': '',
                'airlines': '',
                'arrives_next_day': ''
            },
            'Flights_4': {
                'number_of_tickets': '',
                'seating_class': '',
                'origin_airport': '',
                'destination_airport': '',
                'departure_date': '',
                'return_date': '',
                'is_nonstop': '',
                'outbound_departure_time': '',
                'outbound_arrival_time': '',
                'inbound_arrival_time': '',
                'inbound_departure_time': '',
                'price': '',
                'airlines': ''
            },
            
            # Homes domains
            'Homes_1': {
                'area': '',
                'address': '',
                'property_name': '',
                'phone_number': '',
                'furnished': '',
                'pets_allowed': '',
                'rent': '',
                'visit_date': '',
                'number_of_beds': '',
                'number_of_baths': ''
            },
            'Homes_2': {
                'intent': '',
                'area': '',
                'address': '',
                'property_name': '',
                'phone_number': '',
                'has_garage': '',
                'in_unit_laundry': '',
                'price': '',
                'visit_date': '',
                'number_of_beds': '',
                'number_of_baths': ''
            },
            
            # Hotels domains
            'Hotels_1': {
                'destination': '',
                'number_of_rooms': '',
                'check_in_date': '',
                'number_of_days': '',
                'star_rating': '',
                'hotel_name': '',
                'street_address': '',
                'phone_number': '',
                'price_per_night': '',
                'has_wifi': ''
            },
            'Hotels_2': {
                'where_to': '',
                'number_of_adults': '',
                'check_in_date': '',
                'check_out_date': '',
                'rating': '',
                'address': '',
                'phone_number': '',
                'total_price': '',
                'has_laundry_service': ''
            },
            'Hotels_3': {
                'location': '',
                'number_of_rooms': '',
                'check_in_date': '',
                'check_out_date': '',
                'average_rating': '',
                'hotel_name': '',
                'street_address': '',
                'phone_number': '',
                'price': '',
                'pets_welcome': ''
            },
            'Hotels_4': {
                'location': '',
                'number_of_rooms': '',
                'check_in_date': '',
                'stay_length': '',
                'star_rating': '',
                'place_name': '',
                'street_address': '',
                'phone_number': '',
                'price_per_night': '',
                'smoking_allowed': ''
            },
            
            # Media domains
            'Media_1': {
                'title': '',
                'genre': '',
                'subtitles': '',
                'directed_by': ''
            },
            'Media_2': {
                'movie_name': '',
                'genre': '',
                'subtitle_language': '',
                'director': '',
                'actors': '',
                'price': ''
            },
            'Media_3': {
                'title': '',
                'genre': '',
                'subtitle_language': '',
                'starring': ''
            },
            
            # Movies domains
            'Movies_1': {
                'price': '',
                'number_of_tickets': '',
                'show_type': '',
                'theater_name': '',
                'show_time': '',
                'show_date': '',
                'genre': '',
                'street_address': '',
                'location': '',
                'movie_name': ''
            },
            'Movies_2': {
                'title': '',
                'genre': '',
                'aggregate_rating': '',
                'starring': '',
                'director': ''
            },
            'Movies_3': {
                'movie_title': '',
                'genre': '',
                'percent_rating': '',
                'cast': '',
                'directed_by': ''
            },
            
            # Music domains
            'Music_1': {
                'song_name': '',
                'artist': '',
                'album': '',
                'genre': '',
                'year': '',
                'playback_device': ''
            },
            'Music_2': {
                'song_name': '',
                'artist': '',
                'album': '',
                'genre': '',
                'playback_device': ''
            },
            'Music_3': {
                'track': '',
                'artist': '',
                'album': '',
                'genre': '',
                'year': '',
                'device': ''
            },
            
            # Payment domain
            'Payment_1': {
                'payment_method': '',
                'amount': '',
                'receiver': '',
                'private_visibility': ''
            },
            
            # Rental Cars domains
            'RentalCars_1': {
                'type': '',
                'car_name': '',
                'pickup_location': '',
                'pickup_date': '',
                'pickup_time': '',
                'pickup_city': '',
                'dropoff_date': '',
                'total_price': ''
            },
            'RentalCars_2': {
                'car_type': '',
                'car_name': '',
                'pickup_location': '',
                'pickup_date': '',
                'pickup_time': '',
                'pickup_city': '',
                'dropoff_date': '',
                'total_price': ''
            },
            'RentalCars_3': {
                'car_type': '',
                'car_name': '',
                'pickup_location': '',
                'start_date': '',
                'pickup_time': '',
                'city': '',
                'end_date': '',
                'price_per_day': '',
                'add_insurance': ''
            },
            
            # Restaurants domains
            'Restaurants_1': {
                'restaurant_name': '',
                'date': '',
                'time': '',
                'serves_alcohol': '',
                'has_live_music': '',
                'phone_number': '',
                'street_address': '',
                'party_size': '',
                'price_range': '',
                'city': '',
                'cuisine': ''
            },
            'Restaurants_2': {
                'restaurant_name': '',
                'date': '',
                'time': '',
                'has_seating_outdoors': '',
                'has_vegetarian_options': '',
                'phone_number': '',
                'rating': '',
                'address': '',
                'number_of_seats': '',
                'price_range': '',
                'location': '',
                'category': ''
            },
            
            # Ride Sharing domains
            'RideSharing_1': {
                'destination': '',
                'shared_ride': '',
                'ride_fare': '',
                'approximate_ride_duration': '',
                'number_of_riders': ''
            },
            'RideSharing_2': {
                'destination': '',
                'ride_type': '',
                'ride_fare': '',
                'wait_time': '',
                'number_of_seats': ''
            },
            
            # Services domains
            'Services_1': {
                'stylist_name': '',
                'phone_number': '',
                'average_rating': '',
                'is_unisex': '',
                'street_address': '',
                'city': '',
                'appointment_date': '',
                'appointment_time': ''
            },
            'Services_2': {
                'dentist_name': '',
                'phone_number': '',
                'address': '',
                'city': '',
                'appointment_date': '',
                'appointment_time': '',
                'offers_cosmetic_services': ''
            },
            'Services_3': {
                'doctor_name': '',
                'phone_number': '',
                'average_rating': '',
                'street_address': '',
                'city': '',
                'appointment_date': '',
                'appointment_time': '',
                'type': ''
            },
            'Services_4': {
                'therapist_name': '',
                'phone_number': '',
                'address': '',
                'city': '',
                'appointment_date': '',
                'appointment_time': '',
                'type': ''
            },
            
            # Travel domain
            'Travel_1': {
                'location': '',
                'attraction_name': '',
                'category': '',
                'phone_number': '',
                'free_entry': '',
                'good_for_kids': ''
            },
            
            # Weather domain
            'Weather_1': {
                'precipitation': '',
                'humidity': '',
                'wind': '',
                'temperature': '',
                'city': '',
                'date': ''
            },
            
            # Alarm domain
            'Alarm_1': {
                'alarm_time': '',
                'alarm_name': '',
                'new_alarm_time': '',
                'new_alarm_name': ''
            },
            
            # Trains domain
            'Trains_1': {
                'from': '',
                'to': '',
                'from_station': '',
                'to_station': '',
                'date_of_journey': '',
                'journey_start_time': '',
                'total': '',
                'number_of_adults': '',
                'class': '',
                'trip_protection': ''
            },
            
            # Messaging domain
            'Messaging_1': {
                'location': '',
                'contact_name': ''
            }
        },
        'booked': {},
        'request_state': {},
        'terminated': False,
        'history': []
    }
    
    return state


def generate_state_from_ontology(ontology_path=None):
    """
    Generate a default state from the SGD ontology.json file.
    This function dynamically creates the belief state based on the ontology structure.
    
    Args:
        ontology_path (str, optional): Path to the ontology.json file. 
                                     If None, uses the default path.
    
    Returns:
        dict: A default state dictionary with belief state populated from the ontology.
    """
    import json
    import os
    
    if ontology_path is None:
        # Default path to SGD ontology
        ontology_path = os.path.join(os.path.dirname(__file__), 
                                    '..', '..', '..', 
                                    'data', 'sgd', 'data', 'ontology.json')
    
    try:
        with open(ontology_path, 'r') as f:
            ontology = json.load(f)
    except FileNotFoundError:
        # Fallback to the hardcoded state if ontology file is not found
        return default_state()
    
    # Extract state structure from ontology
    state_structure = ontology.get('state', {})
    
    # Create belief state from ontology
    belief_state = {}
    for domain, slots in state_structure.items():
        belief_state[domain] = {}
        for slot in slots:
            belief_state[domain][slot] = ''
    
    # Create the full state structure
    state = {
        'user_action': [],
        'system_action': [],
        'belief_state': belief_state,
        'booked': {},
        'request_state': {},
        'terminated': False,
        'history': []
    }
    
    return state
# User - System action
LIST_USER_ACT = "user_action: affirm_intent, bye, general, inform, negate_intent, select, " \
                "inform_intent, request, thank, request_alts, negate, greet, affirm"
LIST_SYSTEM_ACT_KETOD = "inform, request, confirm, offer, notify_success, notify_failure, inform_count, offer_intent," \
                        "req_more, bye, greet, welcome, general"
LIST_SYSTEM_ACT_FUSED = "inform, request, select, recommend, no_offer, book, no_book, offer_book, offer_booked," \
                        "req_more, bye, greet, welcome, general"

LIST_CHITCHAT_ACT = ["THANK_YOU", "THANK", "GOODBYE", "BYE", "GREET"]
dict_user_action = {'INFORM': 'inform',
                    'REQUEST': 'request',
                    'INFORM_INTENT': 'inform_intent',
                    'NEGATE_INTENT': 'negate_intent',
                    'AFFIRM_INTENT': 'affirm_intent',
                    'AFFIRM': 'affirm',
                    'NEGATE': 'negate',
                    'SELECT': 'select',
                    'THANK_YOU': 'thank',
                    'THANK': 'thank',
                    'GOODBYE': 'bye',
                    'BYE': 'bye',
                    'GREET': 'greet',
                    'GENERAL': 'general',
                    'REQUEST_ALTS': 'request_alts'}

# # Schema-guided
# RentalCar = "RentalCar; Slots: type, car_name, pickup_location, pickup_date, pickup_time, city, end_date, " \
#             "total_price, price_per_day, add_insurance"
# RideSharing = "RideSharing; Slots: destination, shared_ride, ride_fare, approximate_ride_duration, number_of_riders, " \
#               "ride_type, wait_time, number_of_seats"
# Buses = "Buses; Slots: origin, destination, from_station, to_station, departure_date, departure_time, price, " \
#         "num_passengers, transfers, fare_type, additional_luggage, category"
#
# Flight = "Flight; Slots: origin_city, passengers, seating_class, destination_city, origin_airpot, " \
#          "destination_airpot, depature_date, return_date, number_stops, outbound_departure_time, " \
#          "outbound_arrival_time, inbound_arrival_time, inbound_departure_time, price, refundable, airlines, is_nonstop"
# Train = "Train; Slots: from, to, depart, day, leave, price, people, class, trip_protection, arrive, id"
# Taxi = "Taxi; Slots: arrive, car, depart, dest, leave, phone"
# Messaging = "Messaging; Slots: location, contact_name"
# Hotel = "Hotel; Slots: address, number_of_rooms, check_in_date, number_of_days, rating, hotel_name, address, " \
#         "phone_number, price_per_night, price, has_wifi, number_of_adults, check_out_date, pets_welcome, " \
#         "smoky_allows, has_laundry_service"
# Restaurant = "Restaurant; Slots: restaurant_name, date, time, serves_alcohol, has_live_music, phone_number, " \
#              "address, number_of_seats, price_range, city, cuisine, has_seating_outdoors, has_vegetarian_options, " \
#              "rating, postcode"
# Travel = "Travel; Slots: location, attraction_name, category, phone_number, free_entry, good_for_kids"
# Booking = "Booking: Slot: day, people, stay, time, price, name"
# Weather = "Weather; Slots: precipitation, humidity, wind, temperature, city, date"
# HOSPITAL = "Hospital; Slots: Addr, Phone, Post"
# Services = "Services; Slots: stylist_name, phone_number, average_rating, is_unisex, street_address, city, type, " \
#            "appointment_date, appointment_time, denstist_name, offer_cosmetic_services, doctor_name, therapist_name"
# HOME = "Home; Slots: area, address, property_name, phone_number, furnished, pets_allowed, intent, visit_date," \
#        "numer_of_beds, number_of_baths, has-garage, in_unit_laundry, price"
# Media = "Media; Slots: title, genre, subtile, director, actors, price"
# Music = "Music; Slots: song_name, artist, album, genre, year, device"
# Movies = "Movies; Slots: price, number_of_tickets, show_type, theater_name, show_time, show_date, genre, " \
#          "street_address, location, movie_name, aggregate_rating, starring, director"
# Events = "Events; Slots: category, subcategory, event_name, date, time, number_of_seats, city_of_event, " \
#          "event_location, address_of_location, event_type, number_of_tickets, venue, venue_addressm, price_per_ticket"
# Calendar = "Calendar; Slots: event_date, event_time, event_location, event_name, available_start_time, " \
#            "available_end_time "
# ATTRACTION = "Attraction; Slots: addr, area, name, choice, fee, open, phone, post, price, type"
# Banks = "Banks; Slots: account_type, recipient_account_type, balance, amount, recipient_name, transfer_time"
# Payment = "Payment; Slots: payment_method, amount, receiver, private_visibility"
#
# # dictionary of domain and slot
# dict_schema = {'RentalCar': RentalCar,
#                'RideSharing': RideSharing,
#                'Buses': Buses,
#                'Flight': Flight,
#                'Train': Train,
#                'Taxi': Taxi,
#                'Messaging': Messaging,
#                'Hotel': Hotel,
#                'Restaurant': Restaurant,
#                'Travel': Travel,
#                'Booking': Booking,
#                'Weather': Weather,
#                'HOSPITAL': HOSPITAL,
#                'Services': Services,
#                'HOME': HOME,
#                'Media': Media,
#                'Music': Music,
#                'Movies': Movies,
#                'Events': Events,
#                'Calendar': Calendar,
#                'ATTRACTION': ATTRACTION,
#                'Banks': Banks,
#                'Payment': Payment
#                }

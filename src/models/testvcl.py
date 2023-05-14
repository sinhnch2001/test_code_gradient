import re
import ast
def parse_string(label):
    # Tạo một dictionary trống
    elements = label.split(' <domain> ')
    slots = []
    for element in elements:
        parts = element.split(' <slot> ')
        domain = parts[0]
        domain = domain.replace("<domain> ", "")
        for i in range(1, len(parts)):
            slots.append(domain.lower()+"-"+parts[i])
    return slots

a = parse_string("<domain> RENTALCARS_3 <slot> inform-slot6-the 10th <slot> inform-slot3-March 6th <domain> EVENTS_3 <slot> inform-intent-BuyEventTickets")


label = ["restaurant-name-saffron brasserie", "restaurant-area-centre", "restaurant-food-indian", "restaurant-pricerange-expensive"]
predict = ["attraction-name-nusha", "restaurant-name-saffron brasserie", "restaurant-area-centre", "restaurant-food-indian", "restaurant-pricerange-expensive"]

print("label:", label, "\npredict:", predict)
T = set(label) | set(predict)
M = set(label) - set(predict)
W = set(predict) - set(label)
RSA = len(T-M-W)/len(T)
JGA = 1 if set(label) == set(predict) else 0

print("\n T:", T,"\n", "M:", M,"\n", "W:", W,"\n", "RSA:", RSA,"\n", "JGA:", JGA)

# YOUR CODE HERE
N, M = map(int, input().split())
plates = input().split()
special_colors = input().split()
prices = list(map(int, input().split()))

# P_0 is the default price, P_1 to P_M are prices for special colors
default_price = prices[0]
special_prices = prices[1:]

# Create a mapping from special colors to their prices
color_to_price = {}
for i in range(M):
    color_to_price[special_colors[i]] = special_prices[i]

# Calculate total cost
total_cost = 0
for plate_color in plates:
    if plate_color in color_to_price:
        total_cost += color_to_price[plate_color]
    else:
        total_cost += default_price

print(total_cost)
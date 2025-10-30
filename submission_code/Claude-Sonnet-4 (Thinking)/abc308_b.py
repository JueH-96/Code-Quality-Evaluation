# YOUR CODE HERE
# Read N and M
N, M = map(int, input().split())

# Read colors eaten
colors_eaten = input().split()

# Read specific colors
specific_colors = input().split()

# Read prices
prices = list(map(int, input().split()))
P_0 = prices[0]
P_specific = prices[1:]  # P_1, P_2, ..., P_M

# Create a mapping from specific colors to their prices
color_price_map = {}
for i in range(M):
    color_price_map[specific_colors[i]] = P_specific[i]

# Calculate total
total = 0
for color in colors_eaten:
    if color in color_price_map:
        total += color_price_map[color]
    else:
        total += P_0

print(total)
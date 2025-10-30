# Read input
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Create a dictionary mapping colors to prices
price_map = {}
for i in range(M):
    price_map[D[i]] = P[i + 1]

# Calculate the total cost
total_cost = 0
for color in C:
    if color in price_map:
        total_cost += price_map[color]
    else:
        total_cost += P[0]  # Default price for non-special colors

# Print the result
print(total_cost)
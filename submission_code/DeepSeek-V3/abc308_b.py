# YOUR CODE HERE
# Read input values
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Create a dictionary to map D_i to P_i
price_map = {}
for i in range(M):
    price_map[D[i]] = P[i+1]

# Calculate the total price
total = 0
for color in C:
    if color in price_map:
        total += price_map[color]
    else:
        total += P[0]

# Print the result
print(total)
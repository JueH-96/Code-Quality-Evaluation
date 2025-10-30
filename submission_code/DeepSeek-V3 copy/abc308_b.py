# Read input values
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Create a dictionary to map colors to their prices
price_dict = {}
for i in range(M):
    price_dict[D[i]] = P[i+1]

# Calculate the total price
total = 0
for color in C:
    if color in price_dict:
        total += price_dict[color]
    else:
        total += P[0]

# Print the result
print(total)
# YOUR CODE HERE
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Create a dictionary to map plate colors to prices
price_map = {}
for i in range(M):
    price_map[D[i]] = P[i + 1]

# Calculate total cost
total = 0
for plate in C:
    if plate in price_map:
        total += price_map[plate]
    else:
        total += P[0]

print(total)
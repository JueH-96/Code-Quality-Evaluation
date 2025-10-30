# YOUR CODE HERE
# Read N and M
N, M = map(int, input().split())

# Read colors of plates Takahashi ate
C = input().split()

# Read special plate colors
D = input().split()

# Read prices
prices = list(map(int, input().split()))
P_0 = prices[0]

# Create a dictionary for quick lookup
price_dict = {}
for i in range(M):
    price_dict[D[i]] = prices[i + 1]

# Calculate total cost
total = 0
for plate_color in C:
    if plate_color in price_dict:
        total += price_dict[plate_color]
    else:
        total += P_0

print(total)
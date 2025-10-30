import sys

# Read prices R, G, B
line1 = sys.stdin.readline().split()
R = int(line1[0])
G = int(line1[1])
B = int(line1[2])

# Read disliked color C
C = sys.stdin.readline().strip()

# Create a list of available prices
available_prices = []

if C != "Red":
    available_prices.append(R)

if C != "Green":
    available_prices.append(G)

if C != "Blue":
    available_prices.append(B)

# Find the minimum price among available pens
min_price = min(available_prices)

# Print the result
print(min_price)
# Read the input values
R, G, B = map(int, input().split())
C = input().strip()

# Determine which pen prices are allowed
allowed_prices = []
if C != 'Red':
    allowed_prices.append(R)
if C != 'Green':
    allowed_prices.append(G)
if C != 'Blue':
    allowed_prices.append(B)

# Find and print the minimum allowed price
print(min(allowed_prices))
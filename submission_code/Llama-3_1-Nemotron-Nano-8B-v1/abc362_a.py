# Read the input values
R, G, B = map(int, input().split())
C = input().strip()

# Determine the minimum price based on the disliked color
if C == 'Red':
    min_price = min(G, B)
elif C == 'Green':
    min_price = min(R, B)
else:  # Blue
    min_price = min(R, G)

# Output the result
print(min_price)
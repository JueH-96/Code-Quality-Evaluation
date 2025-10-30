# Read the input values
R, G, B = map(int, input().split())
C = input().strip()

# Determine the minimum cost based on the disliked color
if C == 'Red':
    min_cost = min(G, B)
elif C == 'Green':
    min_cost = min(R, B)
else:  # Blue
    min_cost = min(R, G)

# Output the result
print(min_cost)
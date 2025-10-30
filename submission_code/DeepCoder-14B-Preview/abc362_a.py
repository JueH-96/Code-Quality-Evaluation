# Read the costs of the pens
R, G, B = map(int, input().split())

# Read the disliked color
C = input().strip()

# Create a dictionary mapping each color to its cost
pen_costs = {'Red': R, 'Green': G, 'Blue': B}

# Remove the disliked color's cost
del pen_costs[C]

# Find and print the minimum cost among the remaining options
print(min(pen_costs.values()))
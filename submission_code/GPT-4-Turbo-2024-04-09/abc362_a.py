# Reading the input values
R, G, B = map(int, input().split())
C = input().strip()

# Initialize a dictionary to map colors to their respective costs
costs = {
    'Red': R,
    'Green': G,
    'Blue': B
}

# Remove the disliked color from the dictionary
del costs[C]

# Find the minimum cost from the remaining colors
min_cost = min(costs.values())

# Output the minimum cost
print(min_cost)
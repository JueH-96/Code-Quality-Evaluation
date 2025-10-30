# YOUR CODE HERE
R, G, B = map(int, input().split())
C = input()

# Create a dictionary to map colors to their costs
pen_costs = {'Red': R, 'Green': G, 'Blue': B}

# Remove the disliked color from the dictionary
del pen_costs[C]

# Find the minimum cost among the remaining pens
min_cost = min(pen_costs.values())

# Print the minimum cost
print(min_cost)
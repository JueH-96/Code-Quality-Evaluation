# YOUR CODE HERE

# Read the inputs
R, G, B = map(int, input().split())
C = input()

# Create a dictionary to map colors to their costs
costs = {'Red': R, 'Green': G, 'Blue': B}

# Remove the cost of the color Takahashi dislikes
costs.pop(C)

# Print the minimum cost
print(min(costs.values()))
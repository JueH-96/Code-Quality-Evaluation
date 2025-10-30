# Read the costs of the three pen colors
R, G, B = map(int, input().split())

# Read the color that Takahashi dislikes
C = input().strip()

# Determine the minimum cost to buy a pen
if C == "Red":
    min_cost = min(G, B)
elif C == "Green":
    min_cost = min(R, B)
else:  # C is Blue
    min_cost = min(R, G)

# Print the minimum cost
print(min_cost)
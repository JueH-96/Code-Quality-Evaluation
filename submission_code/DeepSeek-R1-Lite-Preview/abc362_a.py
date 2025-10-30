# Read the costs of the pens
R, G, B = map(int, input().split())

# Read the disliked color
C = input().strip()

# Determine the minimum cost based on the disliked color
if C == "Red":
    min_cost = min(G, B)
elif C == "Green":
    min_cost = min(R, B)
else:
    min_cost = min(R, G)

# Print the minimum cost
print(min_cost)
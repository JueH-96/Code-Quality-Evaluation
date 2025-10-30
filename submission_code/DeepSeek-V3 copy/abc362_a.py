# Read the input values for the costs of the pens
R, G, B = map(int, input().split())
# Read the color that Takahashi dislikes
C = input().strip()

# Determine which pen is not available based on the disliked color
if C == "Red":
    # Cannot buy red pen, choose the minimum between green and blue
    min_cost = min(G, B)
elif C == "Green":
    # Cannot buy green pen, choose the minimum between red and blue
    min_cost = min(R, B)
elif C == "Blue":
    # Cannot buy blue pen, choose the minimum between red and green
    min_cost = min(R, G)
else:
    # If the color is not recognized, assume all are available (though per constraints, it's not needed)
    min_cost = min(R, G, B)

# Print the minimum cost
print(min_cost)
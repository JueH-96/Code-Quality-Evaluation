# Read the input values for the costs of the pens
R, G, B = map(int, input().split())
# Read the color that Takahashi dislikes
C = input().strip()

# Determine which pen is not allowed based on the color C
if C == 'Red':
    # Cannot buy red pen, so choose the minimum between green and blue
    min_cost = min(G, B)
elif C == 'Green':
    # Cannot buy green pen, so choose the minimum between red and blue
    min_cost = min(R, B)
elif C == 'Blue':
    # Cannot buy blue pen, so choose the minimum between red and green
    min_cost = min(R, G)
else:
    # If C is not one of the expected colors, assume all are allowed (though per constraints, it's not possible)
    min_cost = min(R, G, B)

# Print the minimum cost
print(min_cost)
# YOUR CODE HERE
# Read the costs of Red, Green, and Blue pens
R, G, B = map(int, input().split())

# Read the disliked color
C = input()

# Determine the minimum cost based on the disliked color
min_cost = 0
if C == "Red":
    # Takahashi cannot buy a red pen.
    # He can choose between a green pen (cost G) and a blue pen (cost B).
    # He will choose the cheaper one.
    min_cost = min(G, B)
elif C == "Green":
    # Takahashi cannot buy a green pen.
    # He can choose between a red pen (cost R) and a blue pen (cost B).
    min_cost = min(R, B)
elif C == "Blue":
    # Takahashi cannot buy a blue pen.
    # He can choose between a red pen (cost R) and a green pen (cost G).
    min_cost = min(R, G)

# Print the minimum amount of money needed
print(min_cost)
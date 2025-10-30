# YOUR CODE HERE
R, G, B = map(int, input().split())
C = input().strip()

if C == "Red":
    min_cost = min(G, B)
elif C == "Green":
    min_cost = min(R, B)
else:  # C == "Blue"
    min_cost = min(R, G)

print(min_cost)
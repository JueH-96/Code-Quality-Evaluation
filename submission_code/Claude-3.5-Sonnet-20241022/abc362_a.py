# Read input
R, G, B = map(int, input().split())
C = input()

# Initialize list of available pen costs
costs = []

# Add costs of pens that can be bought based on disliked color
if C != "Red":
    costs.append(R)
if C != "Green":
    costs.append(G) 
if C != "Blue":
    costs.append(B)

# Print minimum cost from available options
print(min(costs))
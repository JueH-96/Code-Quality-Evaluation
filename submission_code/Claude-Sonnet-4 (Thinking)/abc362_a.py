R, G, B = map(int, input().split())
C = input().strip()

costs = []
if C != "Red":
    costs.append(R)
if C != "Green":
    costs.append(G)
if C != "Blue":
    costs.append(B)

print(min(costs))
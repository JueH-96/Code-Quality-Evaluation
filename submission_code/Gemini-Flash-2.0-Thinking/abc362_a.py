r, g, b = map(int, input().split())
c = input()

costs = []
if c != "Red":
    costs.append(r)
if c != "Green":
    costs.append(g)
if c != "Blue":
    costs.append(b)

print(min(costs))
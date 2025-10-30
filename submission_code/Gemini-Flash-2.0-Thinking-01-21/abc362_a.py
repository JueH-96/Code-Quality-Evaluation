r, g, b = map(int, input().split())
c = input()

if c == "Red":
    min_cost = min(g, b)
elif c == "Green":
    min_cost = min(r, b)
elif c == "Blue":
    min_cost = min(r, g)
else:
    min_cost = 0  # Should not reach here according to problem description

print(min_cost)
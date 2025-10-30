n, x, y, z = map(int, input().split())

if x < y:
    path = list(range(x + 1, y + 1))
else:
    path = list(range(x - 1, y - 1, -1))

print("Yes" if z in path else "No")
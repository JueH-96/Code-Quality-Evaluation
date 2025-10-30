n, x, y, z = map(int, input().split())

if x < y:
    route = list(range(x, y + 1))
elif x > y:
    route = list(range(x, y - 1, -1))

print("Yes" if z in route else "No")
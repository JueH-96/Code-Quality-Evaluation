n, x, y, z = map(int, input().split())
if z >= min(x, y) and z <= max(x, y):
    print("Yes")
else:
    print("No")
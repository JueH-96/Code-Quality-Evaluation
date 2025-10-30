n, x, y, z = map(int, input().split())
if x < y:
    print("Yes" if x <= z <= y else "No")
else:
    print("Yes" if y <= z <= x else "No")
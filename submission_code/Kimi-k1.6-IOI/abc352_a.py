n, x, y, z = map(int, input().split())
if (x < y and x <= z <= y) or (x > y and y <= z <= x):
    print("Yes")
else:
    print("No")
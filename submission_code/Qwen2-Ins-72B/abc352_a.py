n, x, y, z = map(int, input().split())

if (x < y and (z > x and z < y)) or (x > y and (z < x and z > y)):
    print("Yes")
else:
    print("No")
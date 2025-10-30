n, x, y, z = map(int, input().split())

if x < y:
    if x <= z <= y:
        print("Yes")
    else:
        print("No")
else:
    if y <= z <= x:
        print("Yes")
    else:
        print("No")
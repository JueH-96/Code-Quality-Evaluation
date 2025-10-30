x, y = input().split()
x = int(x)
y = int(y)

if y > x:
    diff = y - x
    if diff <= 2:
        print("Yes")
    else:
        print("No")
else: # y < x
    diff = x - y
    if diff <= 3:
        print("Yes")
    else:
        print("No")
x, y = map(int, input().split())

if y > x:
    if y - x <= 2:
        print("Yes")
    else:
        print("No")
elif x > y:
    if x - y <= 3:
        print("Yes")
    else:
        print("No")
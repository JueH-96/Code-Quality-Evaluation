x, y = map(int, input().split())

if (y - x <= 2 and y > x) or (x - y <= 3 and y < x):
    print("Yes")
else:
    print("No")
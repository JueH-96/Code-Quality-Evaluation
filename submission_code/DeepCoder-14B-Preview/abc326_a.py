x, y = map(int, input().split())
if (y > x and (y - x) <= 2) or (y < x and (x - y) <= 3):
    print("Yes")
else:
    print("No")
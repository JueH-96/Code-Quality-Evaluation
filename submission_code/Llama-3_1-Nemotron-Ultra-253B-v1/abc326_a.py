x, y = map(int, input().split())
if y > x:
    print("Yes" if y - x <= 2 else "No")
else:
    print("Yes" if x - y <= 3 else "No")
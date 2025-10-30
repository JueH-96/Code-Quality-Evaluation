x, y = map(int, input().split())
diff = y - x
if diff > 0:
    print("Yes" if diff <= 2 else "No")
else:
    print("Yes" if (x - y) <= 3 else "No")
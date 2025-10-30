x, y = map(int, input().split())
diff = abs(x - y)
if diff <= 2 or (x > y and diff <= 3):
    print("Yes")
else:
    print("No")
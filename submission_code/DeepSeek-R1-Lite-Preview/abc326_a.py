X, Y = map(int, input().split())
diff = Y - X

if (diff > 0 and diff <= 2) or (diff < 0 and abs(diff) <= 3):
    print("Yes")
else:
    print("No")
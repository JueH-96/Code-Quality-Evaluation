a, b, c = map(int, input().split())
if b < c:
    print("Yes" if a < b or a >= c else "No")
else:
    print("Yes" if c <= a < b else "No")
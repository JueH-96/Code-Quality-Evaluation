n, t, a = map(int, input().split())
remaining = n - t - a
if t > a + remaining or a > t + remaining:
    print("Yes")
else:
    print("No")
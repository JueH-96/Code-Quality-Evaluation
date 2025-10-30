n, t, a = map(int, input().split())
r = n - t - a
if t > a + r or a > t + r:
    print("Yes")
else:
    print("No")
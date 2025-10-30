n, t, a = map(int, input().split())
rem = n - t - a
if t > a + rem or a > t + rem:
    print("Yes")
else:
    print("No")
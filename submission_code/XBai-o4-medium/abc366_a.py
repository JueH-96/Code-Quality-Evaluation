n, t, a = map(int, input().split())
r = n - t - a

if t > a + r:
    print("Yes")
elif a > t + r:
    print("Yes")
else:
    print("No")
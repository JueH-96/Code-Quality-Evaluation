n, t, a = map(int, input().split())
remaining = n - t - a
if (t + remaining) < a:
    print("Yes")
elif (a + remaining) < t:
    print("Yes")
else:
    print("No")
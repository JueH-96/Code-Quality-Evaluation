# YOUR CODE HERE
n, t, a = map(int, input().split())
remaining = n - t - a
if t + remaining < a:
    print("No")
elif a + remaining < t:
    print("No")
else:
    print("Yes")
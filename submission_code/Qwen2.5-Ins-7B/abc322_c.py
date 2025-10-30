# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

days = [0] * (n + 1)
for i in range(m):
    days[a[i]] = a[i] - a[i-1] if i > 0 else 0

for i in range(1, n + 1):
    print(days[i])
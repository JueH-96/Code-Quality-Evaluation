# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

a.sort()
min_diff = float('inf')

for i in range(n - k + 1):
    min_diff = min(min_diff, a[i + n - k - 1] - a[i])

print(min_diff)
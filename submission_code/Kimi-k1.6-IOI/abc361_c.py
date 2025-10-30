n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
m = n - k
if m == 0:
    print(0)
else:
    min_diff = float('inf')
    for i in range(n - m + 1):
        current = a[i + m - 1] - a[i]
        if current < min_diff:
            min_diff = current
    print(min_diff)
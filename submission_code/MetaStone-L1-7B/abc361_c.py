n, k = map(int, input().split())
a = list(map(int, input().split()))
m = n - k
a.sort()
min_diff = float('inf')
for i in range(len(a) - m + 1):
    current_diff = a[i + m - 1] - a[i]
    if current_diff < min_diff:
        min_diff = current_diff
print(min_diff)
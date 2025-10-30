n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
m = n - k
min_diff = float('inf')

for i in range(n - m + 1):
    current_diff = a[i + m - 1] - a[i]
    if current_diff < min_diff:
        min_diff = current_diff

print(min_diff)
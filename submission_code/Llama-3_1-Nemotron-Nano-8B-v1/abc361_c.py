n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
target_length = n - k
min_diff = float('inf')

for i in range(k + 1):
    j = i + target_length - 1
    diff = a[j] - a[i]
    if diff < min_diff:
        min_diff = diff

print(min_diff)
n, k = map(int, input().split())
a = list(map(int, input().split()))

sorted_a = sorted(a)
min_range = float('inf')

# Try all windows of size (n-k) in the sorted array
for i in range(k + 1):
    min_val = sorted_a[i]
    max_val = sorted_a[i + n - k - 1]
    min_range = min(min_range, max_val - min_val)

print(min_range)
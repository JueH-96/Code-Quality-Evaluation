import bisect

n, k = map(int, input().split())
medicines = []
for _ in range(n):
    a, b = map(int, input().split())
    medicines.append((a, b))

medicines.sort()
sorted_a = [a for a, b in medicines]
sorted_b = [b for a, b in medicines]

# Compute suffix sums
suffix_sum = [0] * (n + 1)
for i in range(n-1, -1, -1):
    suffix_sum[i] = suffix_sum[i+1] + sorted_b[i]

max_a = sorted_a[-1] if n > 0 else 0
left = 1
right = max_a + 1

while left < right:
    mid = (left + right) // 2
    pos = bisect.bisect_left(sorted_a, mid)
    current_sum = suffix_sum[pos]
    if current_sum <= k:
        right = mid
    else:
        left = mid + 1

print(left)
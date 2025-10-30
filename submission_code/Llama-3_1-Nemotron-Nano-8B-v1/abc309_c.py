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
suffix_sums = [0] * (n + 1)
for i in range(n - 1, -1, -1):
    suffix_sums[i] = suffix_sums[i + 1] + sorted_b[i]

max_a = sorted_a[-1][0] if n > 0 else 0
low, high = 1, max_a + 1
ans = high

while low <= high:
    mid = (low + high) // 2
    idx = bisect.bisect_left(sorted_a, mid)
    sum_b = suffix_sums[idx]
    if sum_b <= k:
        ans = mid
        high = mid - 1
    else:
        low = mid + 1

print(ans)
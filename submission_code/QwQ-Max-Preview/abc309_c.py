import bisect

n, k = map(int, input().split())

medicines = []
for _ in range(n):
    a, b = map(int, input().split())
    medicines.append((a, b))

medicines.sort()

a_list = [a for a, b in medicines]
b_list = [b for a, b in medicines]

# Compute suffix sums
suffix_sums = [0] * (n + 1)
for i in range(n - 1, -1, -1):
    suffix_sums[i] = suffix_sums[i + 1] + b_list[i]

max_a = a_list[-1] if n > 0 else 0
low = 1
high = max_a + 1
ans = high

while low <= high:
    mid = (low + high) // 2
    idx = bisect.bisect_left(a_list, mid)
    total = suffix_sums[idx]
    if total <= k:
        ans = mid
        high = mid - 1
    else:
        low = mid + 1

print(ans)
import bisect

n, K = map(int, input().split())
medicines = []
for _ in range(n):
    a, b = map(int, input().split())
    medicines.append((a, b))

medicines.sort()
a = [x[0] for x in medicines]
b_list = [x[1] for x in medicines]

# Compute suffix sums
suffix_sum = [0] * n
suffix_sum[-1] = b_list[-1]
for i in range(n-2, -1, -1):
    suffix_sum[i] = b_list[i] + suffix_sum[i+1]

max_a = a[-1] if n > 0 else 0
low = 1
high = max_a + 1

while low < high:
    mid = (low + high) // 2
    i = bisect.bisect_left(a, mid)
    if i < n:
        current_sum = suffix_sum[i]
    else:
        current_sum = 0
    if current_sum <= K:
        high = mid
    else:
        low = mid + 1

print(high)
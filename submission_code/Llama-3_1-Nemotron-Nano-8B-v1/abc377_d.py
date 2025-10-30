import bisect

n, m = map(int, input().split())

intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Sort intervals by their right endpoint
intervals.sort(key=lambda x: x[1])

sorted_ri = [ri for li, ri in intervals]
prefix_max = []
current_max = 0
for li, ri in intervals:
    current_max = max(current_max, li)
    prefix_max.append(current_max)

sum_invalid = 0
for r in range(1, m + 1):
    # Find the rightmost index where Ri <= r
    j = bisect.bisect_right(sorted_ri, r) - 1
    if j >= 0:
        sum_invalid += prefix_max[j]

total = m * (m + 1) // 2
print(total - sum_invalid)
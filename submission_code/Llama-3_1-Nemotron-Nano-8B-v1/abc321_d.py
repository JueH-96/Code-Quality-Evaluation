import bisect

n, m, p = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

b_sorted = sorted(b)
prefix = [0] * (m + 1)
for i in range(m):
    prefix[i+1] = prefix[i] + b_sorted[i]

total = 0
for ai in a:
    threshold = p - ai
    count_leq = bisect.bisect_right(b_sorted, threshold)
    sum_leq = prefix[count_leq]
    contrib = sum_leq + ai * count_leq + p * (m - count_leq)
    total += contrib

print(total)
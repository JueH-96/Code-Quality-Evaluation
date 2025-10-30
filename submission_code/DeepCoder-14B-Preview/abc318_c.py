n, d, p = map(int, input().split())
f = list(map(int, input().split()))
f.sort(reverse=True)

prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i-1] + f[i-1]

total = prefix[-1]
k_min = (n + d - 1) // d

min_cost = float('inf')

for k in range(0, k_min + 1):
    num_passes = k * d
    if num_passes >= n:
        sum_covered = total
    else:
        sum_covered = prefix[num_passes]
    cost = k * p + (total - sum_covered)
    if cost < min_cost:
        min_cost = cost

print(min_cost)
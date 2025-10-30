n, d, p = map(int, input().split())
f = list(map(int, input().split()))
f.sort(reverse=True)

prefix = [0] * (n + 1)
for i in range(n):
    prefix[i+1] = prefix[i] + f[i]

k_max = (n + d - 1) // d
min_cost = float('inf')

for k in range(0, k_max + 1):
    passes = k * d
    used = min(passes, n)
    sum_remaining = prefix[n] - prefix[used]
    cost = k * p + sum_remaining
    if cost < min_cost:
        min_cost = cost

print(min_cost)
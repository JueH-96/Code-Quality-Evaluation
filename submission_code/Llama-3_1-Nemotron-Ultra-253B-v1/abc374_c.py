n, *rest = map(int, open(0).read().split())
k = rest[:n]
total = sum(k)
min_max = float('inf')

for mask in range(1 << n):
    sum_s = 0
    for i in range(n):
        if mask & (1 << i):
            sum_s += k[i]
    sum_other = total - sum_s
    current_max = max(sum_s, sum_other)
    if current_max < min_max:
        min_max = current_max

print(min_max)
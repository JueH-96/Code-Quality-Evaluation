n = int(input())
k = list(map(int, input().split()))
total = sum(k)
min_max = float('inf')

for mask in range(1 << n):
    sum_a = 0
    for i in range(n):
        if mask & (1 << i):
            sum_a += k[i]
    sum_b = total - sum_a
    current_max = max(sum_a, sum_b)
    if current_max < min_max:
        min_max = current_max

print(min_max)
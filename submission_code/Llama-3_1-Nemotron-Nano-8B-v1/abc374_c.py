n = int(input())
k = list(map(int, input().split()))
total = sum(k)
min_max = total  # Initialize with the worst case

for mask in range(0, 1 << n):
    sum_a = 0
    for i in range(n):
        if mask & (1 << i):
            sum_a += k[i]
    current_max = max(sum_a, total - sum_a)
    if current_max < min_max:
        min_max = current_max

print(min_max)
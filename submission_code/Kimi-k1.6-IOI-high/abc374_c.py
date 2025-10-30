n = int(input())
k = list(map(int, input().split()))
total = sum(k)
min_max = total  # Initialize with the maximum possible value

for mask in range(1, (1 << n) - 1):
    s = 0
    for i in range(n):
        if mask & (1 << i):
            s += k[i]
    current_max = max(s, total - s)
    if current_max < min_max:
        min_max = current_max

print(min_max)
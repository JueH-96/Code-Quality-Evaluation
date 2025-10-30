n = int(input())
k = list(map(int, input().split()))
total = sum(k)
min_val = total  # Initialize with the maximum possible value

for mask in range(1, 1 << n):
    if mask == (1 << n) - 1:
        continue  # Skip the full set
    s = 0
    for i in range(n):
        if mask & (1 << i):
            s += k[i]
    current_max = max(s, total - s)
    if current_max < min_val:
        min_val = current_max

print(min_val)
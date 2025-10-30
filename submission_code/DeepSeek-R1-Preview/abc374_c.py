n = int(input())
k = list(map(int, input().split()))
total = sum(k)
min_max = float('inf')

for mask in range(1 << n):
    s = 0
    for i in range(n):
        if (mask >> i) & 1:
            s += k[i]
    current_max = max(s, total - s)
    if current_max < min_max:
        min_max = current_max

print(min_max)
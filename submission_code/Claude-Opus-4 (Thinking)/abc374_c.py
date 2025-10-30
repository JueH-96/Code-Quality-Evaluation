n = int(input())
k = list(map(int, input().split()))

min_max = float('inf')

# Try all possible partitions using bit manipulation
for mask in range(1 << n):
    group_a = 0
    group_b = 0
    
    for i in range(n):
        if mask & (1 << i):
            group_a += k[i]
        else:
            group_b += k[i]
    
    max_group = max(group_a, group_b)
    min_max = min(min_max, max_group)

print(min_max)
n, m = map(int, input().split())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Sort intervals by R_i
intervals.sort(key=lambda x: x[1])

count = 0
max_l = 0
i = 0

for r in range(1, m + 1):
    # Add all intervals where R_i <= r
    while i < n and intervals[i][1] <= r:
        max_l = max(max_l, intervals[i][0])
        i += 1
    
    # Valid l values are from max_l + 1 to r
    valid_l_count = max(0, r - max_l)
    count += valid_l_count

print(count)
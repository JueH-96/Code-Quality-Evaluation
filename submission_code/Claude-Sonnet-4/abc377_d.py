n, m = map(int, input().split())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

count = 0
for l in range(1, m + 1):
    # Find the minimum R_i among all intervals [L_i, R_i] where l <= L_i
    min_r = m + 1  # Initialize to a value larger than any possible r
    for li, ri in intervals:
        if l <= li:
            min_r = min(min_r, ri)
    
    # r can be from l to min(min_r - 1, m)
    max_valid_r = min(min_r - 1, m)
    if max_valid_r >= l:
        count += max_valid_r - l + 1

print(count)
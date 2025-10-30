# YOUR CODE HERE
n, m = map(int, input().split())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

count = 0
for l in range(1, m + 1):
    # Find the minimum R_i among all intervals where L_i >= l
    min_r = m + 1
    for L_i, R_i in intervals:
        if L_i >= l:
            min_r = min(min_r, R_i)
    
    # Valid r values are from l to min(min_r - 1, m)
    if min_r > l:
        count += min(min_r - 1, m) - l + 1

print(count)
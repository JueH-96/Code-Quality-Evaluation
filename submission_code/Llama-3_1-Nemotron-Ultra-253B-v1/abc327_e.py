import math

n = int(input())
p = list(map(int, input().split()))

denom = [0.0] * (n + 1)
for k in range(1, n + 1):
    denom[k] = 10.0 * (1.0 - 0.9 ** k)

prev = [-float('inf')] * (n + 1)
for i in range(1, n + 1):
    prev[i] = p[i - 1]

max_r = -float('inf')

for m in range(1, n + 1):
    if m == 1:
        current_max_sum = max(prev[1:n + 1])
    else:
        curr = [-float('inf')] * (n + 1)
        current_max = 0.0
        for i in range(1, n + 1):
            if i >= m:
                curr[i] = current_max + p[i - 1]
            if i >= m - 1:
                current_max = max(current_max, prev[i] * 0.9)
        prev = curr
        current_max_sum = max(prev[m:n + 1]) if m <= n else -float('inf')
    
    if current_max_sum == -float('inf'):
        continue
    
    denom_val = denom[m]
    term1 = current_max_sum / denom_val
    term2 = 1200.0 / math.sqrt(m)
    r = term1 - term2
    if r > max_r:
        max_r = r

print("{0:.15f}".format(max_r))
import bisect

n, q = map(int, input().split())
r = list(map(int, input().split()))

# Sort the reindeer requirements
r.sort()

# Compute prefix sums
prefix_sum = [0]
for i in range(n):
    prefix_sum.append(prefix_sum[-1] + r[i])

# Answer queries
for _ in range(q):
    x = int(input())
    
    # Find the rightmost position where prefix_sum[i] <= x
    idx = bisect.bisect_right(prefix_sum, x) - 1
    print(idx)
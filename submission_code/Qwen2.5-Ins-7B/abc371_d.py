# YOUR CODE HERE
from bisect import bisect_left, bisect_right

n = int(input())
x = list(map(int, input().split()))
p = list(map(int, input().split()))
q = int(input())

# Create a list of cumulative sums
cumulative_sums = [0]
for i in range(n):
    cumulative_sums.append(cumulative_sums[-1] + p[i])

# Process each query
for _ in range(q):
    l, r = map(int, input().split())
    i = bisect_left(x, l)
    j = bisect_right(x, r)
    print(cumulative_sums[j] - cumulative_sums[i])
# YOUR CODE HERE
import sys

def solve():
    n = int(input())
    weights = list(map(int, input().split()))
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + weights[i - 1]
    
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            max_weight = -sys.maxsize
            min_weight = sys.maxsize
            for i in range(0, n, k):
                current_weight = prefix_sum[i + k] - prefix_sum[i]
                max_weight = max(max_weight, current_weight)
                min_weight = min(min_weight, current_weight)
            max_diff = max(max_diff, max_weight - min_weight)
    print(max_diff)

t = int(input())
for _ in range(t):
    solve()
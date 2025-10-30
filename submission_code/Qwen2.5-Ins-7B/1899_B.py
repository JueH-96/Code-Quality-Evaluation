# YOUR CODE HERE
from typing import List

def max_weight_difference(n: int, weights: List[int]) -> int:
    prefix_sum = [0]
    for weight in weights:
        prefix_sum.append(prefix_sum[-1] + weight)
    
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            min_weight = float('inf')
            max_weight = float('-inf')
            for i in range(k):
                start = i
                end = (i + 1) * (n // k)
                weight = prefix_sum[end] - prefix_sum[start]
                min_weight = min(min_weight, weight)
                max_weight = max(max_weight, weight)
            max_diff = max(max_diff, max_weight - min_weight)
    
    return max_diff

t = int(input())
for _ in range(t):
    n = int(input())
    weights = list(map(int, input().split()))
    print(max_weight_difference(n, weights))
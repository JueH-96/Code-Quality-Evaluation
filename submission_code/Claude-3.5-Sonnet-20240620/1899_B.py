# YOUR CODE HERE
def solve():
    n = int(input())
    weights = list(map(int, input().split()))
    
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + weights[i]
    
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            truck_weights = [prefix_sum[i + k] - prefix_sum[i] for i in range(0, n, k)]
            max_diff = max(max_diff, max(truck_weights) - min(truck_weights))
    
    return max_diff

t = int(input())
for _ in range(t):
    print(solve())
import sys

def solve_case():
    n = int(input())
    weights = [int(x) for x in input().split()]
    
    max_diff = 0
    for k in range(1, n+1):
        if n % k == 0:
            truck_weights = [0] * (n // k)
            for i in range(n):
                truck_idx = i // k
                truck_weights[truck_idx] += weights[i]
            max_diff = max(max_diff, max(truck_weights) - min(truck_weights))
    
    return max_diff

t = int(input())
for _ in range(t):
    print(solve_case())
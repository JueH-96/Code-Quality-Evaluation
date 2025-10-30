import sys
import itertools

def main():
    N, K, P = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C = parts[0]
        A = parts[1:]
        plans.append((C, A))
    
    min_cost = float('inf')
    
    # Since K and P are small, we can try all possible combinations
    # of plans up to a certain limit. Given N <= 100, we need a smarter approach.
    # However, with K and P being small, we can use dynamic programming.
    
    # Initialize a DP table where dp[mask] represents the minimum cost to achieve
    # the parameter values represented by mask.
    # mask is a tuple of K elements, each representing the sum of the corresponding parameter.
    
    # Initialize the DP with the starting state (0, 0, ..., 0)
    dp = {}
    dp[tuple([0]*K)] = 0
    
    for C, A in plans:
        new_dp = {}
        for mask, cost in dp.items():
            # Option 1: not take the plan
            if mask not in new_dp or cost < new_dp[mask]:
                new_dp[mask] = cost
            # Option 2: take the plan
            new_mask = tuple([min(mask[j] + A[j], P) for j in range(K)])
            if new_mask not in new_dp or cost + C < new_dp[new_mask]:
                new_dp[new_mask] = cost + C
        dp = new_dp
    
    # Now, find the minimum cost among all masks where all elements are >= P
    target = tuple([P]*K)
    min_cost = float('inf')
    for mask, cost in dp.items():
        if all(mask[j] >= P for j in range(K)):
            if cost < min_cost:
                min_cost = cost
    
    if min_cost != float('inf'):
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()
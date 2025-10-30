# YOUR CODE HERE
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
    
    # Since K and P are small, we can try all possible combinations
    # But with N up to 100, it's not feasible to try all 2^100 combinations
    # So we need a smarter approach
    
    # We can use dynamic programming to keep track of the minimum cost for each possible sum of parameters
    # Initialize a dictionary to store the minimum cost for each state
    # The state is a tuple representing the sum of each parameter
    
    # Initialize the DP with the initial state (0, 0, ..., 0)
    dp = {tuple([0]*K): 0}
    
    for C, A in plans:
        new_dp = {}
        for state, cost in dp.items():
            # Option 1: not take the current plan
            if state not in new_dp or new_dp[state] > cost:
                new_dp[state] = cost
            # Option 2: take the current plan
            new_state = tuple([min(state[j] + A[j], P) for j in range(K)])
            if new_state not in new_dp or new_dp[new_state] > cost + C:
                new_dp[new_state] = cost + C
        dp = new_dp
    
    # Now, find the minimum cost among all states where all parameters are at least P
    target = tuple([P]*K)
    min_cost = float('inf')
    for state, cost in dp.items():
        if all(s >= P for s in state):
            if cost < min_cost:
                min_cost = cost
    
    if min_cost != float('inf'):
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()
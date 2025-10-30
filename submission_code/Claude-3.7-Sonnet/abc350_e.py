import sys
sys.setrecursionlimit(10**6)

def min_expected_cost(n, A, X, Y, memo=None):
    if memo is None:
        memo = {}
    
    if n == 0:
        return 0
    
    if n in memo:
        return memo[n]
    
    # Option 1: Deterministic operation
    cost1 = X + min_expected_cost(n // A, A, X, Y, memo)
    
    # Option 2: Stochastic operation
    expected_cost = 0
    for b in range(1, 7):
        expected_cost += min_expected_cost(n // b, A, X, Y, memo)
    cost2 = Y + expected_cost / 6
    
    memo[n] = min(cost1, cost2)
    return memo[n]

def main():
    N, A, X, Y = map(int, input().split())
    result = min_expected_cost(N, A, X, Y)
    print("{:.15f}".format(result))

if __name__ == "__main__":
    main()
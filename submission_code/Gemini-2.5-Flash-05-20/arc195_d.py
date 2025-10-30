import sys

# Set recursion limit for deeper recursion (needed for larger N in some cases)
# Note: For N=2*10^5, this DP is too slow regardless of recursion limit.
# The solution is for N up to a few hundreds typically.
# sys.setrecursionlimit(10**6) 

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Memoization table for dp(l, r)
    # Using a dictionary for sparse memoization, or a 2D array if indices are dense.
    # Given N can be large, dict is safer if not all states are visited.
    # However, this DP typically visits N^2 states.
    memo = {}

    def dp(l, r):
        # Base case: if the subarray is empty, 0 operations needed
        if l > r:
            return 0
        
        # Check if the result is already memoized
        if (l, r) in memo:
            return memo[(l, r)]

        # Option 1: Delete A[l] by itself
        # Cost: 1 (for the deletion) + dp(l+1, r) (cost for the rest of the subarray)
        min_ops = 1 + dp(l + 1, r)

        # Option 2: Find other occurrences of A[l] and group them
        # Iterate through possible k where A[k] == A[l]
        for k in range(l + 1, r + 1):
            if A[k] == A[l]:
                # Cost for clearing A[l+1...k-1] (elements between A[l] and A[k])
                cost_between = dp(l + 1, k - 1)
                
                # Cost for deleting the A[l]/A[k] group (1 operation)
                # Plus cost for clearing A[k+1...r] (elements after A[k])
                cost_after_group = 1 + dp(k + 1, r)
                
                # Update min_ops with this alternative
                min_ops = min(min_ops, cost_between + cost_after_group)
        
        # Memoize the result before returning
        memo[(l, r)] = min_ops
        return min_ops

    # The problem asks for the minimum operations to make the entire sequence empty,
    # which corresponds to dp(0, N-1)
    print(dp(0, N - 1))

# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()
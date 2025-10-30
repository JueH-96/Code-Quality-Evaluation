import sys
import bisect

# It's good practice to place the main logic in a function.
def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Read N (number of people) and M (budget).
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, M = map(int, line1.split())
        
        # Read transportation costs for N people.
        line2 = sys.stdin.readline()
        if not line2: return
        A = list(map(int, line2.split()))
    except (IOError, ValueError):
        # Gracefully handle empty or invalid input.
        return

    # Case 1: If the total cost is within budget, the limit can be infinite.
    if sum(A) <= M:
        print("infinite")
        return

    # Case 2: The total cost exceeds the budget. Find the maximum limit x.
    # We can use binary search on x because the total subsidy is monotonic.

    # Sort costs for efficient calculation of the total subsidy.
    A.sort()

    # Precompute prefix sums to find the sum of a range of costs in O(1).
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]

    # Binary search for the maximum valid x.
    # The answer x cannot exceed max(A), which is at most 10^9.
    # We set a safe upper bound for the search.
    low = 0
    high = 2 * 10**9 
    ans = 0

    while low <= high:
        x = (low + high) // 2
        
        # Find k, the number of people with cost A_i <= x, using binary search.
        k = bisect.bisect_right(A, x)
        
        # Calculate the total subsidy for the limit x.
        # - For k people with A_i <= x, subsidy is their full cost.
        # - For N-k people with A_i > x, subsidy is capped at x.
        total_subsidy = prefix_sum[k] + (N - k) * x
        
        if total_subsidy <= M:
            # This limit x is feasible. Store it and try for a larger one.
            ans = x
            low = x + 1
        else:
            # This limit x is too high. Try a smaller one.
            high = x - 1
            
    print(ans)

solve()
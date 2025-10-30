import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Check for the "infinite" case.
    # Calculate the total cost if everyone receives their full transportation cost (x effectively infinite).
    total_cost_if_unlimited = sum(A)

    # If this total cost is within the budget M, then x can be infinitely large.
    if total_cost_if_unlimited <= M:
        print("infinite")
        return

    # Step 2: If not "infinite", perform binary search to find the maximum finite x.
    # The function f(x) = sum(min(x, A_i)) is monotonically non-decreasing.
    # We are looking for the maximum x such that f(x) <= M.

    # Define the search space for x.
    # x must be non-negative. So, the lower bound is 0.
    # A_i values are up to 10^9. If x exceeds max(A_i), the total subsidy will be sum(A_i),
    # which we've already determined is > M. Thus, the effective upper bound for x
    # we need to consider is max(A_i) (10^9).
    low = 0
    high = 10**9 # Maximum possible value for any A_i

    ans = 0 # This variable will store the largest valid x found so far.

    while low <= high:
        mid = low + (high - low) // 2 # Calculate mid-point to prevent potential overflow for very large low+high

        current_total_subsidy = 0
        for a_val in A:
            current_total_subsidy += min(mid, a_val)
        
        if current_total_subsidy <= M:
            # If the current total subsidy is within budget M, then 'mid' is a possible answer.
            # We store 'mid' as a potential answer and try to find a larger 'x' in the upper half.
            ans = mid
            low = mid + 1
        else:
            # If the current total subsidy exceeds budget M, then 'mid' is too large.
            # We need to look for a smaller 'x' in the lower half.
            high = mid - 1
            
    # After the binary search, 'ans' will hold the maximum x that satisfies the condition.
    print(ans)

# Call the solve function to run the program.
solve()
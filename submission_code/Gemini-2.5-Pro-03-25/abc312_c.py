# YOUR CODE HERE
import sys
import bisect

# Function to solve the problem
def solve():
    # Read N (number of sellers) and M (number of buyers) from input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read seller minimum prices A_i
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read buyer maximum prices B_i
    B = list(map(int, sys.stdin.readline().split()))

    # Sort the arrays A and B. This allows us to efficiently count
    # the number of sellers/buyers satisfying the price condition using binary search.
    A.sort()
    B.sort()

    # Set up the binary search range for the price X.
    # The minimum possible price is 1, as given by the constraints on A_i, B_i.
    low = 1
    
    # Determine the upper bound for the binary search.
    # Let S(X) be the number of sellers willing to sell at price X (A_i <= X).
    # Let B(X) be the number of buyers willing to buy at price X (B_j >= X).
    # We are looking for the minimum integer X such that S(X) >= B(X).
    # Consider X = max(B) + 1 (if M > 0). For this X, any B_j < X, so B(X) = 0.
    # Since S(X) is always non-negative, S(X) >= B(X) holds.
    # Thus, the minimum X must be less than or equal to max(B) + 1.
    # Since the maximum possible value for B_i is 10^9, the maximum possible value for
    # max(B) + 1 is 10^9 + 1. We use this as a safe upper bound.
    # The constraints state N, M >= 1, so M > 0 is guaranteed.
    high = 10**9 + 1 
    
    # Initialize the answer variable. We store the smallest X found so far that satisfies the condition.
    # Initialize it to a value guaranteed to be larger than any possible valid answer.
    ans = high + 1 

    # Perform binary search on the price X.
    while low <= high:
        # Calculate the midpoint price to check. Use integer division.
        mid = low + (high - low) // 2
        
        # Calculate S(mid): the number of sellers i such that A_i <= mid.
        # bisect_right(A, mid) returns the index `k` such that all A[i] for i < k have A[i] <= mid,
        # and all A[i] for i >= k have A[i] > mid. This index `k` is exactly the count we need.
        s_mid = bisect.bisect_right(A, mid)
        
        # Calculate B(mid): the number of buyers j such that B_j >= mid.
        # This is equivalent to the total number of buyers M minus the number of buyers j such that B_j < mid.
        # bisect_left(B, mid) returns the index `k` such that all B[i] for i < k have B[i] < mid,
        # and all B[i] for i >= k have B[i] >= mid. The count of elements < mid is `k`.
        # So, the count of elements >= mid is M - k.
        b_mid = M - bisect.bisect_left(B, mid)
        
        # Check if the condition S(mid) >= B(mid) is satisfied for the current price mid.
        if s_mid >= b_mid:
            # If the condition holds, 'mid' is a potential answer.
            # Since we are looking for the minimum X, we record 'mid' as a possible answer
            # and try searching for an even smaller X in the range [low, mid-1].
            ans = mid
            high = mid - 1
        else:
            # If the condition does not hold, it means S(mid) < B(mid).
            # Since S(X) is non-decreasing and B(X) is non-increasing, S(X)-B(X) is non-decreasing.
            # To satisfy the condition, we need a larger value of X.
            # So we search in the range [mid+1, high].
            low = mid + 1

    # After the binary search loop terminates, 'ans' holds the minimum integer X satisfying the condition.
    print(ans)

# Call the solve function to execute the logic when the script is run.
solve()
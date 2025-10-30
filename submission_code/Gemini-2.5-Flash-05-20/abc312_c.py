import sys
import bisect

# It's good practice to encapsulate the main logic in a function.
def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())

    # Read seller prices A and buyer prices B
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Sort the arrays for efficient binary search (bisect module)
    A.sort()
    B.sort()

    # Define the check function for binary search
    # This function determines if a given price X satisfies the condition.
    def check(X):
        # Calculate sellers_count: number of A_i such that A_i <= X
        # bisect_right returns an insertion point which comes after (to the right of)
        # any existing entries of X. This index is effectively the count of elements <= X.
        sellers_count = bisect.bisect_right(A, X)

        # Calculate buyers_count: number of B_i such that B_i >= X
        # bisect_left returns an insertion point which comes before (to the left of)
        # any existing entries of X. Elements from this index to the end are >= X.
        buyers_count = M - bisect.bisect_left(B, X)

        # The condition is sellers_count >= buyers_count
        return sellers_count >= buyers_count

    # Set up the binary search range
    # The minimum possible price is 1.
    low = 1
    # The maximum possible A_i or B_i is 10^9.
    # An answer could potentially be 10^9 + 1 (e.g., if all A_i=10^9+1, but A_i <= 10^9 in constraints).
    # For safety and to ensure the `check` function will eventually return True,
    # 10^9 + 1 is a suitable upper bound. At X = 10^9 + 1, S(X) will be N, and B(X) will be 0, so N >= 0 is always true.
    high = 10**9 + 1
    
    # Initialize the answer with the highest possible value,
    # which we know satisfies the condition (it's an upper bound).
    ans = high

    # Perform binary search
    while low <= high:
        mid = low + (high - low) // 2 # Calculate mid to prevent overflow for very large low/high

        if check(mid):
            # If mid satisfies the condition, it's a potential answer.
            # We try to find a smaller X, so we store mid and search in the lower half.
            ans = mid
            high = mid - 1
        else:
            # If mid does not satisfy the condition, it means mid is too small.
            # We need a larger X, so we search in the upper half.
            low = mid + 1

    # Print the final minimum X found
    sys.stdout.write(str(ans) + "
")

# Call the solve function to run the program
solve()
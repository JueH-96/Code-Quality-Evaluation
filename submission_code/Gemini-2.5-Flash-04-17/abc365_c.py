import sys

# Use faster I/O
input = sys.stdin.readline

def check(x, N, M, A):
    """
    Checks if the total subsidy is within budget M for a given limit x.
    Args:
        x (int): The proposed subsidy limit.
        N (int): The number of people.
        M (int): The total budget.
        A (list[int]): The list of transportation costs.
    Returns:
        bool: True if the total subsidy for limit x is <= M, False otherwise.
    """
    current_subsidy = 0
    for a in A:
        # The subsidy for person i is min(x, A_i)
        current_subsidy += min(x, a)
        # Optimization: if subsidy already exceeds budget, no need to continue
        # This also prevents potential overflow if using fixed-size integer types
        # (though Python's int handles arbitrary size, it's good practice and saves time).
        if current_subsidy > M:
            return False
    return True # If loop finishes, total subsidy <= M


# Read input
line1 = input().split()
N = int(line1[0])
M = int(line1[1])
A = list(map(int, input().split()))

# Check for infinite case
# The total subsidy is sum(min(x, A_i)). As x increases, min(x, A_i) approaches A_i.
# The maximum possible total subsidy is sum(A_i).
# If this maximum possible subsidy is within the budget M, then the subsidy limit x
# can be made arbitrarily large without exceeding the budget.
# Calculate sum of A. Use Python's sum which handles large integers.
sum_a = sum(A)

if sum_a <= M:
    print("infinite")
else:
    # Finite case: The sum of A_i is greater than M, so setting x very high
    # (specifically, x >= max(A_i)) results in total subsidy > M.
    # This means the maximum possible subsidy limit x must be finite.
    # If the maximum answer x > max(A_i), then S(x) = sum(min(x, A_i)).
    # Since all A_i <= max(A_i), if x > max(A_i), then min(x, A_i) = A_i for all i.
    # So S(x) = sum(A_i).
    # If sum(A_i) > M (which is true in the finite case), then S(x) > M,
    # which means x cannot be the answer.
    # Therefore, if sum(A_i) > M, the maximum answer x must be <= max(A_i).
    # Since max(A_i) <= 10^9, the maximum possible answer x is at most 10^9.
    # We can binary search for the maximum non-negative integer x.
    # The search range can be [0, 10^9 + 1].
    # Using 10^9 + 1 as the upper bound for the search space ensures
    # we cover all possible integer answers up to 10^9 and one value above.

    low = 0
    high = 10**9 + 1 # Search space for the subsidy limit x is integers in [0, 10^9+1].
                     # The maximum possible answer is 10^9 when finite.
                     # Searching slightly above 10^9 guarantees we find the maximum if it is 10^9.
    ans = 0 # Initialize with 0, as check(0) is always True (S(0) = 0 <= M because M >= 1).

    # Binary search for the largest integer `mid` in the range [low, high]
    # such that check(mid) is True.
    while low <= high:
        # Calculate mid point. Use integer division.
        # The formula low + (high - low) // 2 is preferred over (low + high) // 2
        # to avoid potential overflow if low and high are very large, although
        # in Python with arbitrary size integers, this is not strictly necessary,
        # but it's a good habit.
        mid = low + (high - low) // 2
        
        # Check if the current subsidy limit `mid` is valid
        if check(mid, N, M, A):
            # `mid` works as a subsidy limit. It's a potential answer.
            # We want the maximum possible x, so we record `mid` as the best
            # answer found so far and try searching in the upper half for a larger answer.
            ans = mid # mid is a valid answer
            low = mid + 1 # Try searching for a larger answer in the range [mid + 1, high]
        else:
            # `mid` does not work as a subsidy limit (total subsidy > M).
            # `mid` is too large, so the maximum answer must be smaller than `mid`.
            # We discard `mid` and the values above it, and search in the lower half.
            # Search in the range [low, mid - 1].
            high = mid - 1

    # After the loop finishes (low > high), `ans` holds the largest `mid`
    # for which check(mid) was True. This is the maximum possible subsidy limit.
    print(ans)
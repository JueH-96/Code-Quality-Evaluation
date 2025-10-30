import sys
from collections import Counter

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())
    # Read array A (0-indexed internally)
    A = list(map(int, sys.stdin.readline().split()))

    # Use Counter to store frequencies of numbers.
    # Counter is efficient for arbitrary integer keys and handles sparse counts well.
    counts = Counter(A)

    # Find initial mex (smallest non-negative integer not in A)
    # The mex of an array of size N is always <= N.
    # We find the smallest k >= 0 such that counts[k] == 0.
    mex = 0
    while counts[mex] > 0:
        mex += 1

    # Process queries
    for _ in range(Q):
        # Read query: index (1-based) and new value
        i_k, x_k = map(int, sys.stdin.readline().split())
        # Adjust index to be 0-based
        i_k -= 1
        
        old_value = A[i_k]
        new_value = x_k
        
        # Update the array element
        A[i_k] = new_value
        
        # Decrease the count of the old value
        counts[old_value] -= 1
        
        # If the count of old_value becomes zero, it means old_value is now potentially missing.
        # If old_value is less than the current mex, it means old_value is now the
        # smallest missing non-negative integer.
        if counts[old_value] == 0:
             # If old_value > mex, it becoming missing doesn't affect the fact
             # that the current mex (which is smaller) is still missing.
             # If old_value <= mex, and its count is now 0, it means old_value
             # was not the mex before (since counts[old_value] was > 0, otherwise
             # mex would have been old_value or smaller), but now
             # it is potentially the smallest missing number. The smallest possible
             # new mex is old_value.
            if old_value < mex:
                mex = old_value

        # Increase the count of the new value
        counts[new_value] += 1

        # If the new value is equal to the current mex, it means the current mex
        # is now present. The mex must increase to find the next smallest
        # non-negative integer that is missing.
        # This check is valid because if new_value is not equal to the current mex,
        # the current mex is either still missing (its count was 0 and new_value != mex)
        # or the mex decreased due to old_value becoming missing.
        if new_value == mex:
            # Increment mex until we find a value whose count is zero.
            # The loop starts from the old mex value (which is now present).
            # Since the mex is always <= N for an array of size N,
            # the loop will eventually find a missing value <= N + 1 (if 0..N are present).
            # However, the problem constraints and logic imply mex <= N is sufficient.
            while counts[mex] > 0:
                mex += 1

        # Print the mex after the update
        print(mex)

solve()
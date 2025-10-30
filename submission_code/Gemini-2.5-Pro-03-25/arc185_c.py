# YOUR CODE HERE
import sys
from collections import defaultdict
from bisect import bisect_right
# import io, os # Use standard sys.stdin/stdout for compatibility

# # If using PyPy, potentially faster I/O might be needed for large inputs:
# # input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline

def solve():
    """
    Reads input, finds a triplet (i, j, k) such that 1 <= i < j < k <= N
    and A[i-1] + A[j-1] + A[k-1] == X, and prints it. If no such triplet
    exists, prints -1.
    """
    # Read N (number of elements) and X (target sum) from input
    N, X = map(int, sys.stdin.readline().split())
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Create a dictionary to store the indices for each value present in A.
    # Keys will be the values from A, and values will be lists of their
    # 1-based indices in the original sequence. The lists will naturally
    # be sorted as we process the array from left to right.
    value_indices = defaultdict(list)
    for i in range(N):
        # Store 1-based index (i + 1)
        value_indices[A[i]].append(i + 1)

    # Iterate through all possible indices for the first element 'i'
    # We use 1-based indexing (1 to N) for i, j, k to match the problem's output format.
    for i in range(1, N + 1):
        val_i = A[i-1] # Get the value A[i] (using 0-based index for accessing list A)

        # Optimization attempt: if val_i is already too large
        # A[i] + A[j] + A[k] = X. Since A[j]>=1 and A[k]>=1, A[i] must be <= X-2.
        # If val_i >= X, the sum will definitely exceed X.
        # We can use a slightly tighter bound: if val_i >= X - 1, continue
        # (because the smallest possible sum would be val_i + 1 + 1 = val_i + 2)
        # Let's keep it simple for now, the main loop structure dominates complexity.

        # Iterate through all possible indices for the second element 'j'
        # Start j from i + 1 to ensure the condition i < j
        for j in range(i + 1, N + 1):
            val_j = A[j-1] # Get the value A[j]

            # Calculate the sum of the first two elements
            current_sum_ij = val_i + val_j

            # Optimization attempt: if val_i + val_j is already too large
            # A[i] + A[j] + A[k] = X. Since A[k] >= 1, A[i] + A[j] must be <= X-1.
            # If current_sum_ij >= X, the sum will definitely exceed X.
            if current_sum_ij >= X:
                 # Continue to the next 'j'. We cannot break the inner loop because
                 # A is not sorted, so a later 'j' might result in a smaller sum
                 # when combined with the current 'i'.
                 continue

            # Calculate the target value needed for the third element 'k'
            target_k_val = X - current_sum_ij

            # The value A[k] must be positive (>= 1 as per constraints).
            # If target_k_val is not positive, it cannot correspond to an element in A.
            if target_k_val <= 0:
                continue

            # Check if the required third value (target_k_val) exists in the input array A.
            # We check if it's a key in our value_indices dictionary.
            if target_k_val in value_indices:
                
                # If the target value exists, get the list of its 1-based indices.
                # This list is sorted in ascending order.
                possible_k_indices = value_indices[target_k_val]
                
                # We need to find an index 'k' from this list such that k > j
                # (to satisfy the condition i < j < k).
                # We can use binary search (specifically, bisect_right from the bisect module)
                # to efficiently find the first index in possible_k_indices
                # that is strictly greater than j.

                # bisect_right(a, x) returns an insertion point for x in list a.
                # This point is such that all elements a[:pos] are <= x,
                # and all elements a[pos:] are > x.
                pos = bisect_right(possible_k_indices, j)

                # If the calculated insertion point 'pos' is less than the length of the list,
                # it means there is at least one index k in possible_k_indices such that k > j.
                # The smallest such index k is the element at possible_k_indices[pos].
                if pos < len(possible_k_indices):
                    # We found a valid index k > j.
                    k = possible_k_indices[pos]
                    
                    # We have successfully found a triplet (i, j, k) satisfying all conditions:
                    # 1. 1 <= i < j < k <= N
                    # 2. A[i-1] + A[j-1] + A[k-1] == X (equivalent to val_i + val_j + target_k_val == X)
                    print(f"{i} {j} {k}")
                    
                    # Since we only need to print one such triplet, we can exit the function.
                    return

    # If the loops complete without finding any valid triplet that satisfies the conditions
    print("-1")

# Execute the main function to solve the problem
solve()
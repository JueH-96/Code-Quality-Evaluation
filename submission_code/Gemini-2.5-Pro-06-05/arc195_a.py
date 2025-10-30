import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right

def solve():
    """
    Solves the 'At Least Two Subsequences' problem by finding and comparing
    the leftmost and rightmost possible subsequences.
    """
    
    # Use fast I/O
    readline = sys.stdin.readline
    
    try:
        # Read problem size
        N, M = map(int, readline().split())
        
        # Read sequences A and B
        A = list(map(int, readline().split()))
        B = list(map(int, readline().split()))
    except (IOError, ValueError):
        # Gracefully handle empty input lines which can occur in some test environments
        return

    # Step 1: Pre-process A to map each value to a sorted list of its indices.
    # This enables efficient lookups using binary search.
    A_indices = defaultdict(list)
    for i, val in enumerate(A):
        A_indices[val].append(i)

    # Step 2: Compute the indices of the "leftmost" subsequence.
    # This is done by greedily picking the earliest available index for each element of B.
    leftmost_indices = []
    current_a_idx = -1  # Search starts from before the first element of A
    for b_val in B:
        # Check if the required value exists in A at all.
        if b_val not in A_indices:
            print("No")
            return
        
        possible_indices = A_indices[b_val]
        
        # Find the first index in `possible_indices` that is strictly greater than `current_a_idx`.
        # `bisect_right` finds the insertion point, which corresponds to the index
        # of the first element in the sorted list that is > `current_a_idx`.
        it = bisect_right(possible_indices, current_a_idx)
        
        # If the insertion point is at the end of the list, no suitable index was found.
        if it == len(possible_indices):
            print("No")  # Cannot form even one subsequence
            return
            
        # A suitable index is found. Update state for the next iteration.
        next_a_idx = possible_indices[it]
        leftmost_indices.append(next_a_idx)
        current_a_idx = next_a_idx

    # If the loop completes, at least one subsequence (the leftmost one) exists.

    # Step 3: Compute the indices of the "rightmost" subsequence.
    # This is done by greedily picking the latest available index, iterating through B from right to left.
    rightmost_indices = [0] * M
    current_a_idx = N  # Search starts from after the last element of A
    for i in range(M - 1, -1, -1):
        b_val = B[i]
        
        # We know b_val exists in A_indices from the successful leftmost computation.
        possible_indices = A_indices[b_val]
        
        # Find the last index in `possible_indices` that is strictly less than `current_a_idx`.
        # `bisect_left` finds the insertion point for `current_a_idx`. The element at
        # index `it - 1` is the largest element in the list that is < `current_a_idx`.
        it = bisect_left(possible_indices, current_a_idx)
        
        # A suitable index must exist because the leftmost subsequence provides a valid candidate.
        next_a_idx = possible_indices[it - 1]
        rightmost_indices[i] = next_a_idx
        current_a_idx = next_a_idx

    # Step 4: Compare the index lists of the leftmost and rightmost subsequences.
    # If they are different, we have found two distinct subsequences.
    # If they are identical, this is the only way to form the subsequence.
    if leftmost_indices != rightmost_indices:
        print("Yes")
    else:
        print("No")

# Run the solution
solve()
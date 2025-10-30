import sys
from bisect import bisect_left, bisect_right

# Function to solve the problem
def solve():
    # Read input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Build value_to_indices map
    # value_to_indices[val] will be a sorted list of indices where A[i] == val
    value_to_indices = {}
    for i in range(N):
        if A[i] not in value_to_indices:
            value_to_indices[A[i]] = []
        value_to_indices[A[i]].append(i)

    # Calculate first_indices (greedy forward search)
    # first_indices[k] will store the index in A used for B[k] in the first match
    first_indices = [-1] * M
    current_a_index = -1 # Start search for B[0] from index 0 (i.e., > -1)
    for k in range(M):
        target_val = B[k]
        
        # If the value from B is not even present in A, B cannot be a subsequence
        if target_val not in value_to_indices:
            print("No")
            return
            
        indices_list = value_to_indices[target_val]
        
        # Find the smallest index i in indices_list such that i > current_a_index
        # This is the first index >= current_a_index + 1
        pos = bisect_left(indices_list, current_a_index + 1)
        
        # If pos == len(indices_list), it means all remaining indices for target_val
        # are less than or equal to current_a_index, so we cannot find a match after it.
        # B is not a subsequence.
        if pos == len(indices_list):
            print("No")
            return
        
        first_indices[k] = indices_list[pos]
        current_a_index = first_indices[k]

    # If we reached here, at least one subsequence (the first match) exists.

    # Calculate last_indices (greedy backward search)
    # last_indices[k] will store the index in A used for B[k] in the last match
    last_indices = [-1] * M
    current_a_index = N # Start search for B[M-1] from index N-1 (i.e., < N)
    for k in range(M - 1, -1, -1):
        target_val = B[k]
        # We already know target_val is in A from the first_indices calculation.
        indices_list = value_to_indices[target_val]
        
        # Find the largest index i in indices_list such that i < current_a_index
        # This is the largest index <= current_a_index - 1
        pos = bisect_right(indices_list, current_a_index - 1)
        
        # If pos == 0, it means all indices for target_val are >= current_a_index.
        # This case should not occur if the first match succeeded, as the existence
        # of a subsequence guarantees the backward greedy search can also find one.
        # If it somehow occurs, it might imply an issue, but for this problem,
        # if the first match is found, the last match should also be found.
        # So we can safely assume pos > 0 if first match succeeded.
        
        last_indices[k] = indices_list[pos - 1]
        current_a_index = last_indices[k]

    # Compare first_indices and last_indices
    found_difference = False
    for k in range(M):
        if first_indices[k] != last_indices[k]:
            found_difference = True
            break

    if found_difference:
        print("Yes")
    else:
        print("No")

# Call the solve function
solve()
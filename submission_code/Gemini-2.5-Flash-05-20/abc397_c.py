import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the array A
    A = list(map(int, sys.stdin.readline().split()))

    # --- Step 1: Compute prefix_distinct_counts ---
    # prefix_distinct_counts[k] will store the number of distinct elements
    # in the subarray A[0...k] (0-indexed).
    prefix_distinct_counts = [0] * N
    seen_elements_left = set()
    current_distinct_count = 0
    for k in range(N):
        if A[k] not in seen_elements_left:
            seen_elements_left.add(A[k])
            current_distinct_count += 1
        prefix_distinct_counts[k] = current_distinct_count

    # --- Step 2: Compute suffix_distinct_counts ---
    # suffix_distinct_counts[k] will store the number of distinct elements
    # in the subarray A[k...N-1] (0-indexed).
    suffix_distinct_counts = [0] * N
    seen_elements_right = set()
    current_distinct_count = 0
    # Iterate from the end of the array backwards
    for k in range(N - 1, -1, -1):
        if A[k] not in seen_elements_right:
            seen_elements_right.add(A[k])
            current_distinct_count += 1
        suffix_distinct_counts[k] = current_distinct_count

    # --- Step 3: Find the maximum sum of distinct counts ---
    # The problem statement defines `i` as the 1-indexed split point.
    # For a split at `i` (1 <= i <= N-1):
    # - First subarray: (A_1, ..., A_i) which is A[0...i-1] in 0-indexed Python.
    # - Second subarray: (A_{i+1}, ..., A_N) which is A[i...N-1] in 0-indexed Python.
    # We iterate `split_idx` from 1 to N-1. This `split_idx` directly corresponds
    # to the problem's 1-indexed `i`.

    max_total_distinct = 0

    # Iterate through all possible split points
    # `split_idx` serves as the 0-indexed starting position of the second subarray,
    # and also as the 1-indexed length of the first subarray (problem's `i`).
    for split_idx in range(1, N): # split_idx goes from 1 to N-1
        # Distinct count for the first subarray (A[0...split_idx-1])
        # This is obtained from prefix_distinct_counts at index `split_idx - 1`.
        left_distinct = prefix_distinct_counts[split_idx - 1]
        
        # Distinct count for the second subarray (A[split_idx...N-1])
        # This is obtained from suffix_distinct_counts at index `split_idx`.
        right_distinct = suffix_distinct_counts[split_idx]
        
        current_total_distinct = left_distinct + right_distinct
        max_total_distinct = max(max_total_distinct, current_total_distinct)

    # Print the final answer
    print(max_total_distinct)

# Call the solve function to execute the program
solve()
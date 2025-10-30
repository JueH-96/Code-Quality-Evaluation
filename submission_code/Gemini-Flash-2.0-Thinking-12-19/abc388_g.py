import sys

# Set recursion depth limit if necessary (not needed for this problem structure)
# sys.setrecursionlimit(2000)

# Read input N
N = int(sys.stdin.readline())

# Read input array A
# Use map and split for efficient reading
A = list(map(int, sys.stdin.readline().split()))

# Read input Q
Q = int(sys.stdin.readline())

# Process each query
results = []
for _ in range(Q):
    # Read query range (L, R)
    L, R = map(int, sys.stdin.readline().split())

    # Adjust L and R to 0-based indexing for Python list slicing
    # Input L, R are 1-based and inclusive [L, R]
    # Python slice A[start:end] is 0-based and exclusive [start, end)
    # So, the range of indices in A is [L-1, R-1]
    l_idx = L - 1
    r_idx = R # The slice is A[l_idx : r_idx] which corresponds to indices L-1, L, ..., R-1

    # The number of mochi in the range is M = R - L + 1
    M = r_idx - l_idx

    # If there are fewer than 2 mochi, no pair can be formed.
    if M < 2:
        results.append(0)
        continue

    # Apply the two-pointer greedy approach on the sorted subarray A[l_idx : r_idx]
    # We conceptually divide the available mochi in the range into two groups for pairing:
    # The first M // 2 elements are candidates for the smaller 'a' mochi.
    # The remaining M - M // 2 elements are candidates for the larger 'b' mochi.
    # This split ensures that any 'a' candidate comes from an index <= any 'b' candidate's starting index.
    # Indices for 'a' candidates in original A: [l_idx, l_idx + M // 2)
    # Indices for 'b' candidates in original A: [l_idx + M // 2, r_idx)

    p1 = l_idx # Pointer for the current potential 'a' mochi index in A
    p2 = l_idx + M // 2 # Pointer for the current potential 'b' mochi index in A

    count = 0 # Counter for the number of kagamimochi formed

    # The pointer p1 iterates through the indices in the first partition [l_idx, l_idx + M//2)
    # The pointer p2 iterates through the indices in the second partition [l_idx + M//2, r_idx)
    # The loop continues as long as there are elements left to consider as 'a' (in the first partition)
    # and elements left to consider as 'b' (in the second partition).
    
    # The upper limit for p1 is the start of the second partition (exclusive).
    p1_limit = l_idx + M // 2

    while p1 < p1_limit and p2 < r_idx:
        # A[p1] is the current 'a' candidate (smallest available from the first partition)
        # A[p2] is the current 'b' candidate being checked (smallest available from the second partition >= p2)
        
        # We need A[p2] >= 2 * A[p1]
        # Use 2 * A[p1] directly; Python handles large integers.
        required_b = 2 * A[p1]

        if A[p2] < required_b:
            # A[p2] is too small to be a partner for A[p1].
            # Since A is sorted, any element before A[p2] in the second partition (if we somehow considered them)
            # would also be too small or already used.
            # So, we need a larger mochi for A[p1]. We move to the next potential 'b' mochi in the second partition.
            p2 += 1
        else: # A[p2] >= required_b
            # Found a valid partner A[p2] for A[p1].
            # We form a pair (A[p1], A[p2]). This pair uses distinct mochi because p1 < p2 (since p1 < p1_limit <= p2).
            count += 1

            # A[p1] is used as 'a', A[p2] is used as 'b'.
            # We move to the next potential 'a' candidate. According to the greedy strategy
            # using the smallest available 'a', the next candidate is A[p1+1],
            # which is the next element in the first partition.
            p1 += 1
            # We also move to the next potential 'b' candidate.
            # Since A[p2] is used, the next potential 'b' must be strictly after A[p2].
            # We start searching for the partner for A[p1+1] from the element after A[p2].
            p2 += 1
            
    # Append the result for the current query
    results.append(count)

# Print the results for all queries
for res in results:
    print(res)
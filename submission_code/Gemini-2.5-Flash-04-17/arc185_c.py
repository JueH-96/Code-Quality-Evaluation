import sys

# Use faster input reading
input = sys.stdin.readline

# Read N and X
N, X = map(int, input().split())

# Read the array A
# Adjust to 0-based indexing internally
A = list(map(int, input().split()))

# Iterate through all possible first elements A[i] (0-indexed i)
# The first index i must be from 0 to N-3 (since we need i < j < k and minimum k is i+2)
for i in range(N - 2):
    # The target sum for the remaining two elements A[j] and A[k] is X - A[i]
    target_jk = X - A[i]

    # Use a hash map to store elements in the suffix A[i+1 ... N-1] and their indices.
    # We iterate j from i+1 to N-2. For each A[j], we need A[k] = target_jk - A[j]
    # where k is in the range [j+1, N-1].
    # To efficiently find such k, we pre-process the range A[i+1 ... N-1].
    # We are specifically looking for k > j.
    # Let's build a map storing value -> latest index from the right in A[i+1 ... N-1].
    # When we iterate j from i+1 to N-2, if we find required_k_val = target_jk - A[j]
    # in this map, the stored index k_candidate is the rightmost index in A[i+1 ... N-1]
    # where required_k_val occurs. If k_candidate > j, it means we found an element A[k_candidate]
    # after A[j] that satisfies the sum condition.

    val_to_last_idx = {}
    # Iterate from the right end of the potential (j, k) search space [i+1, N-1]
    # and store the rightmost index for each value encountered.
    # The loop range is N-1 down to i+1. Indices in A are 0 to N-1.
    # So iterate m from N-1 down to i+1 (inclusive).
    for m in range(N - 1, i, -1): # m goes from N-1 down to i+1
        val = A[m]
        # Store the index m if this value hasn't been seen yet in the scan from the right.
        # This ensures that for any value, val_to_last_idx[val] holds the largest index m
        # in A[i+1 ... N-1] where val appears.
        if val not in val_to_last_idx:
             val_to_last_idx[val] = m

    # Iterate through possible second elements A[j] (0-indexed j)
    # The second index j must be from i+1 to N-2
    for j in range(i + 1, N - 1):
        # We need A[k] = target_jk - A[j]
        required_k_val = target_jk - A[j]

        # required_k_val must be at least 1 (since A[k] >= 1)
        if required_k_val < 1:
            continue

        # Check if the required value exists in the suffix A[j+1 ... N-1]
        # We built val_to_last_idx for A[i+1 ... N-1].
        # If required_k_val is in the map, val_to_last_idx[required_k_val] gives
        # the rightmost index k_candidate >= i+1 where required_k_val occurs.
        if required_k_val in val_to_last_idx:
            k_candidate = val_to_last_idx[required_k_val]

            # Check if this candidate index k_candidate is indeed greater than j.
            # If k_candidate is the rightmost index in A[i+1 ... N-1], and k_candidate > j,
            # it means there exists A[k_candidate] such that A[i] + A[j] + A[k_candidate] = X
            # and i < j < k_candidate. This is a valid triple.
            if k_candidate > j:
                # Found a solution. Print 1-based indices and exit.
                print(i + 1, j + 1, k_candidate + 1)
                sys.exit()

# If the loops complete without finding a solution, print -1.
print(-1)
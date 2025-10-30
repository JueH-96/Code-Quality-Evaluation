import sys

# Read N
N = int(sys.stdin.readline())

# Read A
# A is guaranteed to be sorted in ascending order
A = list(map(int, sys.stdin.readline().split()))

# Read Q
Q = int(sys.stdin.readline())

# Process each query
for _ in range(Q):
    # Read L and R (1-indexed) for the query range
    L, R = map(int, sys.stdin.readline().split())

    # Convert L and R to 0-indexed for Python list access
    # The sublist for the query is A[L-1]...A[R-1]
    l_idx = L - 1
    r_idx = R - 1

    # Length of the sublist A[l_idx...r_idx]
    M = r_idx - l_idx + 1

    # Constraint 1 <= L_i < R_i <= N implies R_i - L_i >= 1, so M >= 2.
    # Thus, the sublist always has at least 2 elements and pairs are possible.
    # No need to explicitly check M < 2 for 0 pairs based on constraints.

    # Apply the two-pointer greedy strategy on the sublist A[l_idx...r_idx]
    # This strategy finds the maximum number of pairs by attempting to pair
    # elements from the first half of the sublist with elements from the second half.
    # The first half contains the first floor(M/2) elements of the sublist.
    # The second half contains the remaining M - floor(M/2) elements of the sublist.

    count = 0

    # Pointer 'i' for elements considered as potential top pieces.
    # These are taken from the first conceptual half of the sublist.
    # 'i' starts at the beginning of the sublist range [l_idx, ..., r_idx]
    i = l_idx

    # Pointer 'j' for elements considered as potential bottom pieces.
    # These are taken from the second conceptual half of the sublist.
    # 'j' starts at the beginning of the second conceptual half.
    j = l_idx + M // 2

    # Iterate while both pointers are within their respective parts of the sublist range
    # 'i' covers indices from l_idx up to (l_idx + M // 2 - 1)
    # 'j' covers indices from (l_idx + M // 2) up to r_idx
    while i < l_idx + M // 2 and j <= r_idx:
        # Check if A[i] (the current potential top piece from the first half)
        # can be a top piece for A[j] (the current potential bottom piece from the second half).
        # The condition for stacking is A[i] <= A[j] / 2, which is equivalent to A[i] * 2 <= A[j]
        # (since A_i >= 1, A[i] * 2 will not overflow standard integer types in Python).
        if A[i] * 2 <= A[j]:
            # A[i] can be paired with A[j]. Form this pair.
            count += 1

            # A[i] has been used as a top piece. Move pointer 'i' to consider the next potential top piece
            # from the first half.
            i += 1

            # A[j] has been used as a bottom piece. Move pointer 'j' to consider the next potential bottom piece
            # from the second half.
            j += 1
        else:
            # A[i] is too large to be a top piece for A[j].
            # A[j] cannot be the bottom piece for the current A[i].
            # Since A is sorted, A[j] is the smallest element currently considered from the second half.
            # If A[i] is too large for A[j], it will also be too large for any A[k] with k < j (within the second half).
            # We need a potentially larger potential bottom piece for A[i].
            # Keep pointer 'i' as A[i] might be able to be paired with a larger element from the second half later.
            # Move pointer 'j' to consider the next potential bottom piece from the second half.
            j += 1

    # Print the maximum number of pairs found for this query using this greedy strategy
    print(count)
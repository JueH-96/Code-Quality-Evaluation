import sys

# Function to calculate mex(a, b, c)
# mex(S) is the minimum non-negative integer that is not in the set S.
# Here, a, b, c are restricted to values 0, 1, or 2.
def calculate_mex(a, b, c):
    s = {a, b, c}
    if 0 not in s:
        return 0
    if 1 not in s:
        return 1
    if 2 not in s:
        return 2
    # If 0, 1, and 2 are all present in the set {a, b, c}, then mex is 3.
    return 3

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = sys.stdin.readline().strip()

    # Precompute suffix_X_counts
    # suffix_X_counts[val][k] stores the count of indices 'idx' such that:
    #   1. idx >= k (0-indexed)
    #   2. S[idx] == 'X'
    #   3. A[idx] == val
    # This array is of size 3x(N+1). The N+1 dimension allows for easy lookup of suffix counts from index (j+1).
    # The last column (N) will effectively be all zeros, serving as a base case for the recursion.
    suffix_X_counts = [[0] * (N + 1) for _ in range(3)]

    # Iterate from N-1 down to 0 to build suffix counts efficiently
    for k in range(N - 1, -1, -1):
        # Copy counts from the next index (k+1) to the current index (k) for all values
        for val in range(3):
            suffix_X_counts[val][k] = suffix_X_counts[val][k + 1]
        # If the character at the current index k is 'X',
        # increment the count for its corresponding A value at this index k.
        if S[k] == 'X':
            suffix_X_counts[A[k]][k] += 1

    total_mex_sum = 0
    
    # current_M_counts[val] stores the count of indices 'i' such that:
    #   1. i < j (where 'j' is the current iteration index in the main loop)
    #   2. S[i] == 'M'
    #   3. A[i] == val
    current_M_counts = [0] * 3

    # Iterate through the sequence with 'j' as the middle index for 'E'
    for j in range(N):
        if S[j] == 'M':
            # If S[j] is 'M', it contributes to 'current_M_counts' for future 'E's (i.e., for j' > j)
            current_M_counts[A[j]] += 1
        elif S[j] == 'E':
            # If S[j] is 'E', this is our middle element (A_j).
            # We need to find 'M's to the left (i < j) and 'X's to the right (k > j).
            Aj = A[j] # The value A[j] for the 'E' element at index j

            # Iterate through all 3x3 (9) possible combinations for the values of A_i (Ai_val)
            # and A_k (Ak_val).
            for Ai_val in range(3): # Possible value for A[i] where S[i] == 'M'
                for Ak_val in range(3): # Possible value for A[k] where S[k] == 'X'
                    # Get the number of 'M's with value Ai_val located to the left of j
                    num_M = current_M_counts[Ai_val]
                    # Get the number of 'X's with value Ak_val located to the right of j
                    # We use j + 1 because k must be strictly greater than j.
                    num_X = suffix_X_counts[Ak_val][j + 1]

                    # Only proceed if there are actual 'M' and 'X' elements for this value combination
                    # to avoid unnecessary calculations.
                    if num_M > 0 and num_X > 0:
                        # Calculate the mex value for the triplet (Ai_val, Aj, Ak_val)
                        mex_val = calculate_mex(Ai_val, Aj, Ak_val)
                        # Add the product of mex_val, count of M's, and count of X's to the total sum.
                        # This effectively sums mex_val for all num_M * num_X such triplets.
                        total_mex_sum += mex_val * num_M * num_X
        # If S[j] == 'X', no action is needed here in the main loop.
        # Its count is already handled by the `suffix_X_counts` precomputation.

    print(total_mex_sum)

# Call the main solver function
solve()
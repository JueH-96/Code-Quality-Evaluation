import sys

def solve():
    # Read N and K from the first line
    N, K = map(int, sys.stdin.readline().split())
    # Read the array A from the second line
    A = list(map(int, sys.stdin.readline().split()))

    # Maximum possible value for any A_i based on constraints (1 <= A_i <= 10^6)
    MAX_A_VAL = 10**6 

    # --- Step 1: Precompute counts of each number and their original indices ---
    # `counts[val]` stores the frequency of `val` in A.
    # E.g., if A = [3, 4, 3], counts[3] will be 2.
    counts = [0] * (MAX_A_VAL + 1)
    
    # `pos_in_A[val]` stores a list of original indices where `A[idx] == val`.
    # This allows us to quickly find all occurrences of a specific value.
    # E.g., if A = [3, 4, 3], pos_in_A[3] will be [0, 2].
    pos_in_A = [[] for _ in range(MAX_A_VAL + 1)]

    for idx, val in enumerate(A):
        counts[val] += 1
        pos_in_A[val].append(idx)

    # --- Step 2: Precompute `multiples_count[g]` ---
    # `multiples_count[g]` stores the total number of elements in A that are multiples of `g`.
    # This is done by iterating from largest possible value down to 1.
    # For a given `g`, we sum `counts[m]` for all `m` that are multiples of `g`.
    # E.g., multiples_count[2] will include counts[2], counts[4], counts[6], etc.
    multiples_count = [0] * (MAX_A_VAL + 1)
    
    # The calculation is efficient: O(MAX_A_VAL * log(MAX_A_VAL)).
    for g in range(MAX_A_VAL, 0, -1):
        for m in range(g, MAX_A_VAL + 1, g):
            multiples_count[g] += counts[m]

    # --- Step 3: Determine the maximum GCD for each A_i ---
    # `answers[i]` will store the final maximum GCD for A_i.
    # Initialize all answers to 1, as 1 is the smallest possible GCD.
    answers = [1] * N
    
    # `locked[idx]` is a boolean flag. If `locked[idx]` is True, it means
    # `answers[idx]` has been assigned its final (maximum) GCD. This prevents
    # redundant assignments and ensures that each answer index is processed
    # at most once for its final GCD, leading to overall O(N) updates.
    locked = [False] * N
    
    # `locked_count` tracks how many `answers` have been finalized.
    # Used for an early exit optimization.
    locked_count = 0 

    # Iterate `g` from `MAX_A_VAL` down to `1`.
    # This order is crucial: by checking larger `g` values first, the first `g`
    # found for a given `A_i` will automatically be its maximum possible GCD.
    for g in range(MAX_A_VAL, 0, -1):
        # If `multiples_count[g] >= K`, it means there are at least K elements in A
        # that are multiples of `g`. Thus, `g` is a possible GCD for a K-subset.
        # Any `A_i` that is a multiple of `g` could potentially have `g` as its answer.
        if multiples_count[g] >= K:
            # Now, for all `A_i` that are multiples of `g`, `g` is a candidate GCD.
            # We iterate through all multiples `m` of `g`.
            for m in range(g, MAX_A_VAL + 1, g):
                # If `m` actually exists in the original array A (i.e., `pos_in_A[m]` is not empty)
                if pos_in_A[m]:
                    # For each original index `idx` where `A[idx]` was equal to `m`
                    for idx in pos_in_A[m]:
                        # If `answers[idx]` has not yet been finalized (i.e., `locked[idx]` is False)
                        if not locked[idx]:
                            answers[idx] = g      # Assign `g` as the current max GCD for A[idx]
                            locked[idx] = True    # Mark this index as finalized
                            locked_count += 1     # Increment the count of finalized answers
            
            # Optimization: If all N answers have been finalized, we can break early
            # from the outer loop, as no smaller `g` can yield a better GCD for any A_i.
            if locked_count == N:
                break

    # --- Step 4: Print the results ---
    # Print each calculated answer on a new line.
    for ans in answers:
        sys.stdout.write(str(ans) + '
')

# This is the standard way to run the script when executed
if __name__ == '__main__':
    solve()
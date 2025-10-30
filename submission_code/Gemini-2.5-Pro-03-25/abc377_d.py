# YOUR CODE HERE
import sys

def solve():
    # Read N and M from input
    N, M = map(int, sys.stdin.readline().split())
    
    # `minR[k]` will store the minimum R_i among all intervals [L_i, R_i] such that L_i = k.
    # Initialize with a value larger than M. M+1 is sufficient because all R_i <= M.
    # Array size is M+1 to use 1-based indexing (indices 1 to M). Index 0 is unused.
    minR = [M + 1] * (M + 1)
    
    # Read N intervals [L_i, R_i]
    for _ in range(N):
        L, R = map(int, sys.stdin.readline().split())
        # Constraints guarantee 1 <= L <= R <= M.
        # Update the minimum R value seen so far for intervals starting at L.
        # If multiple intervals start at the same L, we only need the one with the smallest R
        # for the purpose of calculating the suffix minimum later.
        minR[L] = min(minR[L], R)

    # `Rmin_vals[l]` will store the minimum R_i among all intervals [L_i, R_i] such that L_i >= l.
    # This is computed efficiently using a suffix minimum calculation on the `minR` array.
    # Array size is M+1, index 0 unused. Initialize with 0, but values will be overwritten.
    Rmin_vals = [0] * (M + 1) 
    
    # `current_min` will keep track of the minimum R value encountered so far
    # when iterating from M down to 1. This represents min{ R_i | L_i >= l+1 } when computing for l.
    current_min = M + 1
    
    # Calculate Rmin_vals using suffix minimum logic
    # Iterate l from M down to 1
    for l in range(M, 0, -1):
        # Update `current_min` by considering the minimum R value for intervals starting exactly at l.
        # If no interval starts at l, minR[l] is M+1, which won't decrease `current_min` unless it was already M+1.
        current_min = min(current_min, minR[l])
        # Store the overall minimum R for all intervals starting at l or later.
        # This value is min( minR[k] for k in range(l, M + 1) )
        Rmin_vals[l] = current_min

    # Calculate the total number of pairs (l, r) such that 1 <= l <= r <= M.
    # This is the sum of integers from 1 to M, which is M * (M + 1) / 2.
    total_pairs = M * (M + 1) // 2

    # Calculate the total number of "bad" pairs (l, r).
    # A pair (l, r) is bad if it fully contains at least one interval [L_i, R_i].
    # This means there exists an index i such that l <= L_i and R_i <= r.
    total_bad_pairs = 0
    
    # Iterate through all possible values of the left endpoint l, from 1 to M.
    for l in range(1, M + 1):
        # For a fixed l, a pair (l, r) is bad if there exists an interval [L_i, R_i] such that l <= L_i and R_i <= r.
        # This condition is equivalent to requiring r >= min{R_i | L_i >= l}.
        # The value min{R_i | L_i >= l} has been precomputed and stored in Rmin_vals[l].
        # If no interval [L_i, R_i] exists with L_i >= l, then Rmin_vals[l] will be M+1.
        # In this case, the condition r >= M+1 cannot be satisfied for r <= M, correctly indicating
        # that no interval starting at or after l imposes a constraint making (l, r) bad.
        
        # The conditions for a pair (l, r) to be counted as bad for this fixed l are:
        # 1. 1 <= l <= M (implicit by loop range)
        # 2. l <= r <= M (general condition for pairs (l,r))
        # 3. r >= Rmin_vals[l] (condition for being bad due to some interval [L_i, R_i] with L_i >= l)
        
        # Combine conditions 2 and 3: We need to count r such that max(l, Rmin_vals[l]) <= r <= M.
        
        # Determine the effective lower bound for r for a bad pair starting at l.
        lower_bound_r = max(l, Rmin_vals[l])
        
        # Check if there are any valid r values satisfying the combined condition.
        # This is true if the lower bound does not exceed M (i.e., lower_bound_r <= M).
        if lower_bound_r <= M:
            # The number of integers in the range [A, B] inclusive is B - A + 1.
            # Count the number of valid r values in [lower_bound_r, M].
            count = M - lower_bound_r + 1
            # Add this count to the total number of bad pairs found so far.
            total_bad_pairs += count

    # The number of "good" pairs (satisfying the problem conditions) is the
    # total number of pairs minus the total number of bad pairs.
    ans = total_pairs - total_bad_pairs
    
    # Print the final answer.
    print(ans)

# Execute the solve function to read input, compute, and print the answer.
solve()
# YOUR CODE HERE
import sys

def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())

    # min_R_for_L_val[x] will store the minimum R_j among all given intervals (L_j, R_j)
    # where L_j is equal to x.
    # Initialize all values with M + 1. This value acts as an "infinity" since all R_j <= M.
    # We use 1-based indexing for L_i values, so the array size is M+1 to cover indices from 1 to M.
    min_R_for_L_val = [M + 1] * (M + 1)

    # Process each input interval (L_i, R_i)
    for _ in range(N):
        L_i, R_i = map(int, sys.stdin.readline().split())
        # Update min_R_for_L_val[L_i] if the current R_i is smaller than the previously recorded minimum for L_i.
        min_R_for_L_val[L_i] = min(min_R_for_L_val[L_i], R_i)

    # min_R_suffix[x] will store the minimum R_j among all given intervals (L_j, R_j)
    # where L_j is greater than or equal to x. This also includes the conceptual M+1 bound.
    # Formally, min_R_suffix[x] = min({R_j | L_j >= x} union {M+1}).
    # This array is computed using a suffix minimum approach.
    # The array size is M+2 to allow for indexing up to M+1 (e.g., min_R_suffix[M+1] as a base case).
    min_R_suffix = [M + 1] * (M + 2) 

    # Base case for the suffix minimum: min_R_suffix[M+1] is M+1.
    # This represents the 'M+1' in the union {M+1}, ensuring that if no interval restricts 'r',
    # 'r' can go up to M.
    # Iterate from M down to 1 to compute the suffix minimums.
    # min_R_suffix[x] is the minimum of min_R_for_L_val[x] and min_R_suffix[x+1].
    for x in range(M, 0, -1):
        min_R_suffix[x] = min(min_R_for_L_val[x], min_R_suffix[x+1])

    total_count = 0
    # Iterate through all possible starting points 'l' for the interval [l, r]
    for l in range(1, M + 1):
        # For a fixed 'l', an interval [l, r] is valid if for ALL given intervals [L_i, R_i],
        # [l, r] does not completely contain [L_i, R_i].
        # This condition is equivalent to (l > L_i OR R_i > r) for all i.
        
        # If l > L_i, the condition is satisfied for this i, regardless of r.
        # If l <= L_i, then we MUST have R_i > r for this i (i.e., r < R_i).
        # Therefore, for a fixed 'l', 'r' must be strictly less than R_i for all intervals [L_i, R_i] where L_i >= l.
        # This implies that 'r' must be less than the minimum of all such R_i values (including M+1).
        # The term min({R_j | L_j >= l} union {M+1}) is exactly what min_R_suffix[l] stores.
        
        # So, valid 'r' values must satisfy: l <= r < min_R_suffix[l].
        # The maximum possible 'r' for the current 'l' is (min_R_suffix[l] - 1).
        r_upper_bound = min_R_suffix[l] - 1
        
        # The number of valid 'r' values for the current 'l' is calculated as (r_upper_bound - l + 1).
        # We use max(0, ...) to handle cases where r_upper_bound is less than l, which means no valid 'r' exists.
        num_r_for_l = max(0, r_upper_bound - l + 1)
        
        total_count += num_r_for_l

    # Print the final total count of valid (l, r) pairs.
    print(total_count)

# Call the solve function to execute the program
solve()
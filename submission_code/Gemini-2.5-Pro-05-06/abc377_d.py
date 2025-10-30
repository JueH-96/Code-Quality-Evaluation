import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    # val[i] stores min R_k where L_k = i.
    # We use 1-based indexing for problem coordinates L, R.
    # Python list val will have size M+1, so indices 0..M. val[0] is unused.
    # Initialize with M+1, which acts as infinity since all R_k <= M.
    val = [M + 1] * (M + 1) 
    for _ in range(N):
        L_i, R_i = map(int, sys.stdin.readline().split())
        # L_i is 1-based index. Constraints: 1 <= L_i <= R_i <= M.
        val[L_i] = min(val[L_i], R_i)

    # min_R_ge_l[i] stores min R_k for any original interval (L_k, R_k)
    # where L_k >= i. This is S_i from the thought process.
    # Python list min_R_ge_l will have size M+2, indices 0..M+1.
    # min_R_ge_l[0] is unused.
    # min_R_ge_l[M+1] is the base case for suffix minimums, representing infinity.
    min_R_ge_l = [M + 1] * (M + 2) 
    # min_R_ge_l[M+1] is already M+1.

    # Calculate suffix minimums. min_R_ge_l[i] effectively stores min(val[i], val[i+1], ..., val[M]).
    for i in range(M, 0, -1): # i from M down to 1
        min_R_ge_l[i] = min(val[i], min_R_ge_l[i+1])

    num_bad_pairs = 0
    # Iterate over all possible l_coord values for pairs (l_coord,r)
    for l_coord in range(1, M + 1): # l_coord from 1 to M
        # For a fixed l_coord, a pair (l_coord, r) is "bad" if it contains some [L_k,R_k].
        # This means l_coord <= L_k and R_k <= r for some k.
        # This is equivalent to r >= min {R_j | L_j >= l_coord}.
        # This minimum is exactly min_R_ge_l[l_coord].
        # Let this minimum be actual_min_R_for_this_l.
        actual_min_R_for_this_l = min_R_ge_l[l_coord]
        
        # So, for (l_coord, r) to be bad, r must be >= actual_min_R_for_this_l.
        # Also, any valid pair must satisfy l_coord <= r (problem constraint).
        # Combining these, r must be at least max(l_coord, actual_min_R_for_this_l).
        lower_bound_for_r = max(l_coord, actual_min_R_for_this_l)
        
        # Also, any valid pair must satisfy r <= M (problem constraint).
        # So, bad r's range from lower_bound_for_r to M, inclusive.
        if lower_bound_for_r <= M:
            # Number of such r values is M - lower_bound_for_r + 1.
            num_bad_pairs += (M - lower_bound_for_r + 1)
            
    total_possible_pairs = M * (M + 1) // 2
    ans = total_possible_pairs - num_bad_pairs
    
    sys.stdout.write(str(ans) + "
")

solve()
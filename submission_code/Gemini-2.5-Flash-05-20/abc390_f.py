import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # pos_min[v]: smallest index (0-indexed) where value v occurs in A. N if not present.
    # pos_max[v]: largest index (0-indexed) where value v occurs in A. -1 if not present.
    pos_min = [N] * (N + 1)
    pos_max = [-1] * (N + 1)

    for i in range(N):
        val = A[i]
        pos_min[val] = min(pos_min[val], i)
        pos_max[val] = max(pos_max[val], i)

    # calc_pairs_optimized(N_arr, L_max_limit, R_min_limit)
    # Counts (L, R) pairs such that 0 <= L <= R <= N_arr - 1
    # and L <= L_max_limit and R >= R_min_limit.
    def calc_pairs_optimized(N_arr, L_max_limit, R_min_limit):
        # L_max_limit or R_min_limit might be out of range, e.g., -1 or N_arr.
        # Handle cases where the effective range for L or R is empty.
        if L_max_limit < 0 or R_min_limit >= N_arr:
            return 0

        count = 0

        # Part 1: L < R_min_limit. Here, max(L, R_min_limit) is R_min_limit.
        # L iterates from 0 up to min(L_max_limit, R_min_limit - 1)
        L_range_end_1 = min(L_max_limit, R_min_limit - 1)
        if L_range_end_1 >= 0: # Ensure the range for L is valid
            num_Ls_1 = L_range_end_1 - 0 + 1 # Number of L values in this part (0-indexed)
            count += num_Ls_1 * (N_arr - R_min_limit) # Each L gets (N_arr - R_min_limit) valid R values

        # Part 2: L >= R_min_limit. Here, max(L, R_min_limit) is L.
        # L iterates from max(0, R_min_limit) up to L_max_limit
        L_range_start_2 = max(0, R_min_limit)
        if L_max_limit >= L_range_start_2: # Ensure the range for L is valid
            num_terms_2 = L_max_limit - L_range_start_2 + 1 # Number of L values in this part
            # Sum of (N_arr - 1 - L + 1) = (N_arr - L) for L from L_range_start_2 to L_max_limit
            # This is an arithmetic series sum.
            first_term_2 = N_arr - L_range_start_2
            last_term_2 = N_arr - L_max_limit
            count += num_terms_2 * (first_term_2 + last_term_2) // 2
        
        return count

    total_sum_f_LR = 0

    # Term 1: sum_{x=1}^N count_present(x)
    # This equals sum_{L=1}^N sum_{R=L}^N |S_{L,R}| (sum of distinct elements)
    sum_of_distinct_counts = 0
    for x in range(1, N + 1):
        if pos_max[x] == -1: # Value x is not present in A
            count_x_present = 0
        else:
            count_x_present = calc_pairs_optimized(N, pos_max[x], pos_min[x])
        sum_of_distinct_counts += count_x_present
    
    total_sum_f_LR += sum_of_distinct_counts

    # Term 2: - sum_{x=2}^N count_both_present(x, x-1)
    # This subtracts the cases where x and x-1 are both present in S_{L,R}, and x-1 would not be a segment start.
    sum_of_connections_to_subtract = 0
    for x in range(2, N + 1):
        if pos_max[x] == -1 or pos_max[x-1] == -1: # Either x or x-1 is not present
            count_x_and_x_minus_1_present = 0
        else:
            # For x and x-1 to both be present in A[L..R], L must be <= min(pos_max[x], pos_max[x-1])
            # and R must be >= max(pos_min[x], pos_min[x-1]).
            L_max_joint = min(pos_max[x], pos_max[x-1])
            R_min_joint = max(pos_min[x], pos_min[x-1])
            count_x_and_x_minus_1_present = calc_pairs_optimized(N, L_max_joint, R_min_joint)
        sum_of_connections_to_subtract += count_x_and_x_minus_1_present
    
    total_sum_f_LR -= sum_of_connections_to_subtract

    sys.stdout.write(str(total_sum_f_LR) + "
")

solve()
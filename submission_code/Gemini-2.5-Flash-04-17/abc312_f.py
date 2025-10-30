import sys

# Function to get value from sorted list, handling out-of-bounds by returning 0
# This works because X_i >= 1. A value of 0 effectively means no contribution
# from an item that isn't there when considering marginal gains.
def get_list_value(lst, idx):
    if 0 <= idx < len(lst):
        return lst[idx]
    return 0

def solve():
    # Use faster input
    input = sys.stdin.readline

    N, M = map(int, input().split())
    
    P = [] # Pull-tab cans (T=0)
    R = [] # Regular cans (T=1)
    O = [] # Can openers (T=2)
    
    for _ in range(N):
        T, X = map(int, input().split())
        if T == 0:
            P.append(X)
        elif T == 1:
            R.append(X)
        else:
            O.append(X)

    # Sort lists in descending order for greedy selection
    P.sort(reverse=True)
    R.sort(reverse=True)
    O.sort(reverse=True)

    len_P = len(P)
    len_R = len(R)
    len_O = len(O)

    # Calculate prefix sums for quick sum calculation
    P_prefix_sum = [0] * (len_P + 1)
    for i in range(len_P):
        P_prefix_sum[i+1] = P_prefix_sum[i] + P[i]

    R_prefix_sum = [0] * (len_R + 1)
    for i in range(len_R):
        R_prefix_sum[i+1] = R_prefix_sum[i] + R[i]

    O_prefix_sum = [0] * (len_O + 1)
    for i in range(len_O):
        O_prefix_sum[i+1] = O_prefix_sum[i] + O[i]

    max_total_happiness = 0

    # Iterate over the number of openers selected (k2)
    # k2 can range from 0 up to min(M, len_O)
    for k2 in range(min(M, len_O) + 1):
        # Get the total capacity from the top k2 openers
        current_opener_capacity = O_prefix_sum[k2]

        # Number of remaining items to pick from P U R
        M_prime = M - k2 

        # If we need to pick negative items from P U R, this k2 is invalid
        if M_prime < 0:
            continue

        # Determine the valid range for the number of regular cans selected (n_r)
        # We pick n_r items from R and M_prime - n_r items from P.
        # Constraints:
        # 0 <= n_r <= len_R
        # 0 <= M_prime - n_r <= len_P
        # M_prime - n_r >= 0  => n_r <= M_prime
        # M_prime - n_r <= len_P => n_r >= M_prime - len_P
        # Combining: max(0, M_prime - len_P) <= n_r <= min(M_prime, len_R)
        
        j_min = max(0, M_prime - len_P)
        j_max = min(M_prime, len_R)

        # If the valid range for n_r is empty, continue
        if j_min > j_max:
            continue

        # We need to maximize H(n_r) = P_prefix_sum[M_prime - n_r] + R_prefix_sum[min(n_r, current_opener_capacity)]
        # over the range [j_min, j_max] for n_r.

        # This function is piecewise based on min(n_r, current_opener_capacity).

        # Piece 1: n_r in [j_min, min(j_max, current_opener_capacity)]
        # In this range, H1(n_r) = P_prefix_sum[M_prime - n_r] + R_prefix_sum[n_r]
        # This function is concave. We can find the maximum using binary search
        # for the point where H1(n_r) >= H1(n_r+1).

        upper_bound1 = min(j_max, current_opener_capacity)

        if j_min <= upper_bound1:
            # Binary search for the largest n_r in [j_min, upper_bound1]
            # such that H1(n_r) >= H1(n_r+1) (if n_r < upper_bound1)
            # H1(n_r) - H1(n_r+1) = (P_prefix_sum[M_prime - n_r] - P_prefix_sum[M_prime - n_r - 1]) + (R_prefix_sum[n_r] - R_prefix_sum[n_r + 1])
            # Assuming indices are valid, this is P[M_prime - n_r - 1] - R[n_r]
            # Using get_list_value for boundary safety:
            # get_list_value(P, M_prime - n_r - 1) - get_list_value(R, n_r)

            low_bs, high_bs = j_min, upper_bound1
            # Default optimal n_r in this range is the lower bound
            opt_nr1 = j_min 

            # Binary search for the largest n_r in [j_min, upper_bound1]
            # such that (n_r == upper_bound1) or (H1(n_r) >= H1(n_r + 1))
            while low_bs <= high_bs:
                mid_bs = low_bs + (high_bs - low_bs) // 2
                
                is_peak_at_or_before_mid = False

                if mid_bs == upper_bound1:
                    # This is the last point in the range, it could be the peak
                    is_peak_at_or_before_mid = True
                else:
                    # Check if H1(mid_bs) >= H1(mid_bs + 1)
                    # This is equivalent to get_list_value(P, M_prime - mid_bs - 1) >= get_list_value(R, mid_bs)
                    p_val_contrib = get_list_value(P, M_prime - mid_bs - 1)
                    r_val_contrib = get_list_value(R, mid_bs)

                    if p_val_contrib >= r_val_contrib: 
                         is_peak_at_or_before_mid = True

                if is_peak_at_or_before_mid:
                    # The peak is at or before mid_bs. Record mid_bs as a potential optimal and search lower range.
                    opt_nr1 = mid_bs 
                    high_bs = mid_bs - 1 
                else: # H1(mid_bs) < H1(mid_bs + 1)
                    # Function is strictly increasing at mid_bs. Peak is after mid_bs. Search higher range.
                    low_bs = mid_bs + 1

            # Calculate happiness at the optimal n_r found in this range
            current_happiness1 = P_prefix_sum[M_prime - opt_nr1] + R_prefix_sum[opt_nr1]
            max_total_happiness = max(max_total_happiness, current_happiness1)


        # Piece 2: n_r in [max(j_min, current_opener_capacity + 1), j_max]
        # In this range, H2(n_r) = P_prefix_sum[M_prime - n_r] + R_prefix_sum[current_opener_capacity]
        # Since P_prefix_sum is non-decreasing and M_prime - n_r decreases as n_r increases,
        # P_prefix_sum[M_prime - n_r] is non-increasing as n_r increases.
        # R_prefix_sum[current_opener_capacity] is constant for a fixed k2.
        # Thus, H2(n_r) is non-increasing as n_r increases.
        # The maximum is always at the smallest valid n_r in this range.

        lower_bound2 = max(j_min, current_opener_capacity + 1)
        upper_bound2 = j_max

        if lower_bound2 <= upper_bound2:
             # The optimal n_r in this range is the lower bound
             optimal_nr2 = lower_bound2

             # Calculate happiness
             current_happiness2 = P_prefix_sum[M_prime - optimal_nr2] + R_prefix_sum[current_opener_capacity]
             max_total_happiness = max(max_total_happiness, current_happiness2)

    print(max_total_happiness)

solve()
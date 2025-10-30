import sys
import collections
import math

def solve():
    N, S = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums for the first N terms (0-indexed)
    # _P[k] stores sum(A[0]...A[k-1])
    # _P[0] = 0
    prefix_sums_list = [0] * (N + 1)
    for i in range(N):
        prefix_sums_list[i+1] = prefix_sums_list[i] + A[i]

    total_period_sum = prefix_sums_list[N]

    # Map prefix sum values to a list of indices where they occur.
    # Since A_i >= 1, prefix_sums_list values are strictly increasing,
    # so each list in the defaultdict will typically contain only one index.
    prefix_sum_to_indices = collections.defaultdict(list)
    for i in range(N + 1):
        prefix_sum_to_indices[prefix_sums_list[i]].append(i)

    # We are looking for PS(r) - PS(l) = S, where PS(k) = floor(k/N)*total_period_sum + _P[k % N].
    # This simplifies to:
    # S = q * total_period_sum + (_P[j_mod] - _P[i_mod])
    # where q = floor(r/N) - floor(l/N), j_mod = r % N, i_mod = l % N.
    # q must be non-negative.

    # The term (_P[j_mod] - _P[i_mod]) ranges from -total_period_sum to total_period_sum.
    # So, -total_period_sum <= S - q * total_period_sum <= total_period_sum
    # This implies:
    # S - total_period_sum <= q * total_period_sum <= S + total_period_sum

    # Calculate the range of 'q' to check:
    # Lower bound for q: q_lower_bound = (S - total_period_sum) / total_period_sum
    # Upper bound for q: q_upper_bound = (S + total_period_sum) / total_period_sum

    # Using Python's // for floor division.
    q_lower_bound_int = (S - total_period_sum) // total_period_sum
    q_upper_bound_int = (S + total_period_sum) // total_period_sum

    # Iterate through possible values of q. Add a small buffer to ensure covering all edge cases.
    # The number of q values will be small (at most around 5).
    # q must be non-negative.
    for q in range(max(0, q_lower_bound_int - 2), q_upper_bound_int + 2):
        # Calculate the required residual sum (delta_P = _P[j_mod] - _P[i_mod])
        residual_sum_target = S - q * total_period_sum
        
        # Check if residual_sum_target is within the possible range for _P[j_mod] - _P[i_mod]
        if not (-total_period_sum <= residual_sum_target <= total_period_sum):
            continue

        # For each possible starting modular index i_mod (from 0 to N)
        for i_mod in range(N + 1):
            # We are looking for _P[j_mod] such that _P[j_mod] - _P[i_mod] = residual_sum_target
            # This means _P[j_mod] = _P[i_mod] + residual_sum_target
            target_P_val_for_j_mod = prefix_sums_list[i_mod] + residual_sum_target
            
            # Check if such a _P[j_mod] value exists in our precomputed prefix sums
            if target_P_val_for_j_mod in prefix_sum_to_indices:
                # If it exists, find a corresponding j_mod
                for j_mod in prefix_sum_to_indices[target_P_val_for_j_mod]:
                    # Condition for a non-empty subsequence (r > l)
                    # If q == 0, then floor(r/N) == floor(l/N). For r > l, we need j_mod > i_mod.
                    # If S >= 1 and _P[j_mod] - _P[i_mod] = S, then j_mod must be > i_mod for S to be positive,
                    # unless S=0 and j_mod=i_mod which implies empty sum. But S >= 1 is given.
                    # So for q=0, j_mod > i_mod is required.
                    # If q > 0, then floor(r/N) > floor(l/N) is guaranteed. We can always construct actual l, r
                    # (e.g., l = i_mod, r = q*N + j_mod) such that r > l.
                    if q == 0 and j_mod <= i_mod:
                        continue # This combination does not yield a non-empty subsequence sum
                    
                    # If we reached here, a valid subsequence sum is found.
                    print("Yes")
                    return

    # If no such subsequence found after checking all possibilities
    print("No")

solve()
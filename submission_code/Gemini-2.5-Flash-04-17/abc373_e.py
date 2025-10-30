import sys
import bisect

def solve():
    # Read N, M, K
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    K = int(line1[2])

    # Read A
    A = list(map(int, sys.stdin.readline().split()))

    sum_a = sum(A)
    R_total = K - sum_a

    # Handle N=0 case (although constraints say 1 <= N)
    if N == 0:
        print()
        return

    # Store original indices with votes and sort
    # indexed_A is a list of (value, original_index) tuples sorted by value
    indexed_A = sorted([(A[k], k) for k in range(N)])
    orig_idx_to_sorted_pos = {original_idx: sorted_pos for sorted_pos, (_, original_idx) in enumerate(indexed_A)}

    # Compute prefix sums and suffix sums of sorted vote values
    # PS_global[k] = sum of values indexed_A[0]...indexed_A[k]
    # SS_global[k] = sum of values indexed_A[k]...indexed_A[N-1]
    sorted_A_values = [val for val, _ in indexed_A]
    PS_global = [0] * N
    PS_global[0] = sorted_A_values[0]
    for k in range(1, N):
        PS_global[k] = PS_global[k-1] + sorted_A_values[k]

    SS_global = [0] * N
    SS_global[N-1] = sorted_A_values[N-1]
    for k in range(N-2, -1, -1):
        SS_global[k] = SS_global[k+1] + sorted_A_values[k]

    def get_sum_sorted_values_range(start_idx, end_idx):
        if start_idx > end_idx or start_idx < 0 or end_idx >= N:
            return 0
        # Sum of values indexed_A[start_idx]...indexed_A[end_idx]
        # Using suffix sums SS_global
        # The sum is SS_global[start_idx] - sum(indexed_A[end_idx+1:])
        # sum(indexed_A[end_idx+1:]) is SS_global[end_idx+1] if end_idx+1 < N
        return SS_global[start_idx] - (SS_global[end_idx + 1] if end_idx + 1 < N else 0)

    def can_win(i, additional_votes):
        V = A[i] + additional_votes
        R_rem = R_total - additional_votes

        if R_rem < 0:
            return False

        pos_i = orig_idx_to_sorted_pos[i]

        # Count candidates j!=i with A_j > V
        # Find index p in indexed_A where values are > V
        # bisect_right(indexed_A, (V, N)) finds the first index whose element is strictly greater than (V, N)
        # Elements from this index p to N-1 have value > V. Count = N - p.
        # Note: Using (V, N) as the key guarantees that tuples with value V come before it
        # if their original index is < N, which is always true. So this finds the split point correctly.
        p = bisect.bisect_right(indexed_A, (V, N))
        
        # Candidates with vote > V are indexed p, ..., N-1 in indexed_A
        count_greater = N - p

        # If candidate i's initial vote A[i] was > V, then candidate i is one of those counted.
        # We need count of *other* candidates j!=i.
        # Check A[i] value explicitly because indexed_A might have duplicates
        if A[i] > V:
             count_greater -= 1

        # Count candidates j!=i with A_j <= V that we can make surpass V in the worst case.
        # These are candidates k!=i with indexed_A[k][0] <= V (indices 0...p-1 in indexed_A)
        # We want to make `max_affordable_m` of these surpass V
        # Smallest costs V+1 - value correspond to largest values <=V (among others)
        # Largest values <=V (for j!=i) come from indices p-1, p-2, ..., 0 in indexed_A, skipping pos_i if needed.

        # The number of candidates j!=i with A_j <= V is the number of indices k in [0, p-1] such that k != pos_i.
        # This count is p - (1 if pos_i < p else 0)
        actual_num_leV_others = p - (1 if pos_i < p else 0)

        low_m = 0
        high_m = actual_num_leV_others # Max possible candidates from <=V group (excluding i)
        max_affordable_m = 0 # Max number of candidates <=V (excluding i) we can make surpass V

        # Binary search for max_affordable_m in [0, actual_num_leV_others]
        bs_low_m = 0
        bs_high_m = actual_num_leV_others # Inclusive upper bound

        while bs_low_m <= bs_high_m:
            mid_m = (bs_low_m + bs_high_m) // 2
            if mid_m == 0:
                cost_needed = 0
            else:
                # We need the sum of values of the top mid_m candidates among those j!=i with A_j <= V
                # These correspond to values at indices p-mid_m, p-mid_m+1, ..., p-1 in indexed_A, skipping pos_i if needed.
                
                # Sum of values indexed_A[p-mid_m]...indexed_A[p-1]
                sum_top_m_values_leV_global = get_sum_sorted_values_range(p - mid_m, p - 1)

                sum_values_leV_others = sum_top_m_values_leV_global
                # If pos_i is among the top mid_m values (indices p-mid_m to p-1) and A[i] <= V
                if pos_i < p and pos_i >= p - mid_m:
                     sum_values_leV_others -= indexed_A[pos_i][0]
                
                cost_needed = mid_m * (V + 1) - sum_values_leV_others
                
            if cost_needed <= R_rem:
                max_affordable_m = mid_m # mid_m is achievable, try more
                bs_low_m = mid_m + 1
            else:
                bs_high_m = mid_m - 1 # mid_m is not achievable, try fewer

        # max_affordable_m holds the largest value that passed the condition in the binary search
        
        total_surpass = count_greater + max_affordable_m
        return total_surpass < M

    results = []
    for i in range(N):
        # Binary search for the minimum X in [0, R_total]
        # Find minimum X such that can_win(i, X) is True.
        # Property is monotonic: if can_win(i, X) is True, can_win(i, X+1) is True.
        # Binary search for the first True.
        low = 0
        high = R_total # Inclusive upper bound for the answer

        ans_X = -1 # Default to impossible

        # Check if win is possible at all with R_total additional votes
        if not can_win(i, R_total):
             results.append(-1)
             continue

        # Binary search for the minimum X in [0, R_total] that works.
        # The answer is guaranteed to be in [0, R_total].
        bs_low = 0
        bs_high = R_total # Inclusive upper bound for search range

        while bs_low <= bs_high:
            mid = bs_low + (bs_high - bs_low) // 2
            if can_win(i, mid):
                ans_X = mid # mid is possible, try smaller
                bs_high = mid - 1
            else:
                bs_low = mid + 1 # mid is not possible, need more

        results.append(ans_X)

    print(*results)

solve()
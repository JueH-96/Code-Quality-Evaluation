from collections import Counter
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # 1. Check consistency of sums for (A_i != -1, B_i != -1)
    sums_AB = set()
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            sums_AB.add(A[i] + B[i])

    if len(sums_AB) > 1:
        print("No")
        return

    S_fixed = list(sums_AB)[0] if sums_AB else None

    # 2. Calculate lower bound for S
    S_lower_bound = 0
    for i in range(N):
        if A[i] != -1 and B[i] == -1:
            S_lower_bound = max(S_lower_bound, A[i])
        elif A[i] == -1 and B[i] != -1:
            S_lower_bound = max(S_lower_bound, B[i])
        # For (A_i != -1, B_i != -1), A_i and B_i must be non-negative in the end.
        # Required A'_i is A_i, Required B'_i is B_i. A'_i + B'_i = S implies A_i + B_i = S.
        # A'_i >= 0 means A_i >= 0 (true), B'_i >= 0 means B_i >= 0 (true).
        # Also, A'_i = A_i <= S and B'_i = B_i <= S must hold.
        # So S >= A_i and S >= B_i for (AB) types.
        elif A[i] != -1 and B[i] != -1:
            S_lower_bound = max(S_lower_bound, A[i])
            S_lower_bound = max(S_lower_bound, B[i])


    # 3. Define the check function for a given S
    def check(S_val):
        if S_val < 0:
            return False

        if S_fixed is not None and S_val != S_fixed:
            return False

        if S_val < S_lower_bound:
            return False

        fixed_A_vals = []
        req_exact_A_values = []
        num_A_flexible = 0
        N_minus_minus = 0

        for i in range(N):
            if A[i] != -1:
                fixed_A_vals.append(A[i])
                # If A[i] is a fixed value in A, it must be possible to place it.
                # If it's placed at index j, then A'_j = A[i].
                # We need A'_j + B'_j = S_val, so B'_j = S_val - A[i].
                # For B'_j to be non-negative, S_val >= A[i].
                # This check is covered by S_val >= S_lower_bound calculation which includes max(A_i) for A_i!= -1 and B_i=-1 types,
                # and also max(A_i) for A_i != -1 and B_i != -1 types. Is it all A_i != -1? Yes, if it ends up paired with B_i != -1,
                # or with B_i = -1 (needs S >= A_i), or with B_i = -1 and A_i = -1 (needs S >= A'_i = A_i).
                # So S_val must be >= max(A_i for A_i != -1). This is included in S_lower_bound.
                # Let's add an explicit check just to be sure, although S_lower_bound should cover it.
                if A[i] > S_val:
                     return False

            if A[i] == -1:
                num_A_flexible += 1
            
            if A[i] == -1 and B[i] == -1:
                N_minus_minus += 1

            # Determine the required A' value if this position is not (--) type
            if not (A[i] == -1 and B[i] == -1):
                if A[i] != -1 and B[i] != -1: # AB type
                    # Sum must be S_val (checked by S_val == S_fixed if fixed, or S_val != A[i]+B[i] check below if not fixed)
                    if S_fixed is None and S_val != A[i] + B[i]:
                         return False
                    req_exact_A_values.append(A[i])
                elif A[i] == -1 and B[i] != -1: # -B type
                    # Need S_val >= B[i] (covered by S_val >= S_lower_bound)
                    req_exact_A_values.append(S_val - B[i])
                elif A[i] != -1 and B[i] == -1: # A- type
                    # Need S_val >= A[i] (covered by S_val >= S_lower_bound)
                    req_exact_A_values.append(A[i])

        cnt_fixed = Counter(fixed_A_vals)
        cnt_req_exact = Counter(req_exact_A_values)

        # Check count conditions
        # We need to compare the multiset of values we can form in A' with the multiset of values required in A'.
        # Available multiset M_avail = current_A_fixed (p values) + num_A_flexible values >= 0.
        # Required multiset M_req = req_exact_A (q values) + N_minus_minus values in [0, S].
        # p = len(fixed_A_vals), q = len(req_exact_A_values) = N - N_minus_minus.
        # p + num_A_flexible = N. q + N_minus_minus = N.

        # Conditions from count differences:
        # sum(max(0, count_req[v] - count_fixed[v]) for v in [0, S]) <= num_A_flexible
        # sum(max(0, count_fixed[v] - count_req[v]) for v in [0, S]) <= N_minus_minus

        # The relevant values v for counting are those present in cnt_fixed or cnt_req_exact, and 0.
        relevant_values = set(cnt_fixed.keys()) | set(cnt_req_exact.keys()) | {0}

        pos_diff_sum = 0 # Sum of deficits in fixed_A that must be covered by flexible A for values <= S
        neg_diff_sum = 0 # Sum of excesses in fixed_A that must go into (--) spots for values <= S

        for v in relevant_values:
            # Only consider values v <= S, as values > S are handled or impossible.
            # Fixed A values > S are ruled out.
            # Required A values (A_i or S-B_i) are >= 0. S-B_i <= S if B_i >= 0.
            # So, required exact values <= S are the only ones that matter for counts in [0, S].
            if v > S:
                 # If v > S, cnt_req_exact.get(v, 0) must be 0 based on how req_exact_A is formed for a valid S.
                 # So diff_v = 0 - cnt_fixed.get(v, 0). If this is negative, means cnt_fixed[v] > 0.
                 # This case (fixed_A_vals > S) is handled already.
                 continue # Should not happen if previous check is correct

            diff_v = cnt_req_exact.get(v, 0) - cnt_fixed.get(v, 0)
            pos_diff_sum += max(0, diff_v)
            neg_diff_sum += max(0, -diff_v)

        # Condition 7: Total positive deficits in fixed_A (for values <= S needed exactly) must be covered by num_A_flexible slots.
        if pos_diff_sum > num_A_flexible:
            return False

        # Condition 8: Total negative deficits (excesses) in fixed_A (for values <= S) must be accommodated by N_minus_minus slots.
        if neg_diff_sum > N_minus_minus:
            return False

        # If conditions hold, it is possible to form the multisets.
        return True

    # 4. Determine candidate S values to check
    S_candidates = []
    if S_fixed is not None:
        S_candidates.append(S_fixed)
    else:
        S_candidates.append(S_lower_bound)
        # Add a large S candidate. This helps check if the conditions eventually hold for large S.
        # The critical S values are S_lower_bound and potentially sums A_i + B_j.
        # A value comfortably larger than any possible A_i + B_j for A_i, B_j >= 0 is sufficient.
        # Max A_i, B_i <= 10^9. Max sum approx 2 * 10^9. Add N for safety margin.
        S_large = 2 * 10**9 + 2 * 10**9 + N + 5
        S_candidates.append(S_large)

    # 5. Check candidate S values
    for s in S_candidates:
        if check(s):
            print("Yes")
            return

    # No working S found among candidates
    print("No")

solve()
def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, C, K = map(int, input_data[:4])
    A = list(map(int, input_data[4:]))

    # -------------------------------------
    # Special quick case: if C == 0, then (C*k + A_i) mod M = A_i mod M = A_i.
    # The minimum over i does not depend on k, so the answer is just:
    #   K * min(A).
    # -------------------------------------
    if C == 0:
        print(K * min(A))
        return

    # -------------------------------------
    # Otherwise, let g = gcd(C, M).  The sequence (C*k) mod M repeats with period P = M // g.
    # Our sum index k goes from 0 to K-1.  The total number of terms is K.
    #
    # We can split K into:
    #   K = (full_cycles)*P + (remainder)
    #
    # Then
    #   sum_{k=0..K-1} min_{i} (C*k + A_i) mod M
    # = (full_cycles) * [ sum over one full cycle of k=0..P-1 ] + [ sum over partial remainder ]
    #
    # So the main task is to compute:
    #   cycle_sum = sum_{k=0..P-1} min_{i} ( (C*k) mod M + A_i ) mod M
    #
    # But (C*k) mod M, for k=0..P-1, takes on exactly P distinct values which
    # are multiples of g in the range [0, M-1].  (Because gcd(C,M) = g.)
    #
    # Therefore we can rewrite the cycle sum as:
    #   sum_{x in S} min_i (x + A_i) mod M,
    # where S = { 0, g, 2g, ..., M-g }.  (|S| = P = M//g).
    #
    # Now define val(x) := min_i (x + A_i) mod M.
    # Let x in S, i.e. x = t*g for t in [0..P-1].
    # We want to sum val(t*g) for t=0..P-1.
    # However, note that we need the sum for k=0..P-1, i.e. t=0..P-1,
    # but our final formula is the same if we sum t=1..P (just a shift of index),
    # because { t*g mod M | t=0..P-1 } = { t*g mod M | t=1..P }.  The set is the same.
    #
    # So we will actually compute:
    #   Sum_{t=0..P-1} val(t*g) = Sum_{t=1..P} val(t*g),  (same set of x-values).
    #
    # The function val(x) = min_i( (x + A_i) mod M ).  If we sort A into B,
    # let a0 = B[0] (smallest A).  For a given x:
    #   - If x + a < M, then (x + a) mod M = x + a.
    #   - If x + a >= M, then (x + a) mod M = x + a - M.
    # Let T = M - x.  Then T + x = M.
    #   If a < T, then x + a < M => the mod is x + a.  
    #   If a >= T, then x + a >= M => the mod is x + a - M = a - T.
    #
    # Rewriting val(x) = val(M - T):
    #   val(M - T) = min(
    #       (M - T) + min{ a : a < T },    # if there exists a < T
    #       min{ a : a >= T } - T         # if there exists a >= T
    #   )
    #
    # We define k = number of elements a in B that are < T => k = bisect_left(B, T).
    # - If k=0 => no a < T => val = B[0] - T.
    # - If 0<k<N => val = min((M - T) + B[0], B[k] - T).
    # - If k=N => all a < T => val = (M - T) + B[0].
    #
    # In terms of t*g = x = M-T => T = M - x = M - t*g.
    # We'll let t range over 1..P, so x = t*g, T = M - t*g.
    # Because t*g is in {g, 2g, ..., P*g = M}, T goes from M-g, M-2g, ..., 0.
    # But actually, T in [0..M], but we only sum from t=1..P => T in [M-P*g .. M-g], plus we might consider t=0 => T=M.
    # It's simpler to just define a function val_t(t) = val(t*g) and sum t=1..P.
    #
    # We cannot afford to loop over t=1..P if P can be up to 10^9.  We notice val_t(t)
    # depends on k = bisect_left(B, M - t*g).  That k changes only at points where M - t*g = B[i],
    # i.e. t*g = M - B[i].  So we can do a "sweep" in t by jumping from one breakpoint to the next.
    #
    # We'll implement a piecewise approach:
    #
    # 1) Sort B.
    # 2) Let a0 = B[0], aN = B[N-1].
    # 3) We define P = M // g.  We want sum_{t=1..P} val(t*g).
    # 4) We'll break the range t=1..P into segments where k= bisect_left(B, M - t*g) stays constant.
    #    That condition is "M - t*g is in an interval that doesn't cross any B[i]".
    #    Rearrange: t*g = M - s, s in intervals that don't cross any B[i].
    #    So effectively s goes from 0..M, and we break s at each B[i].
    #    Then t*g = M - s.  So when s \in [B[i], B[i+1]), t*g \in (M - B[i+1], M - B[i}] (± end points).
    #    t in [ (M - B[i+1])/g, (M - B[i])/g ].
    #
    # 5) On each such segment, val_t(t) = "a linear function in t" with slope = -g if 0<k<N,
    #    or simply a line with slope -g in the boundary cases k=0 or k=N.  We can sum an arithmetic series in O(1).
    #
    # Implementation details can be tricky.  A more direct (but still efficient) approach:
    #    - We'll build a sorted list of breakpoints s_i = B[i], plus s_0=0, s_{N}=M (to handle edges).
    #    - For each interval [s_i, s_{i+1}), we know the same k= i.  Actually careful with the boundary...
    #    - Then convert that interval in s to an interval in t:  M - s in [M - s_{i+1}, M - s_i)
    #        => t*g in that range => t in [ ceil((M - s_{i+1})/g), floor((M - s_i - 1)/g ) ] etc.
    #    - Evaluate the formula for val(t*g) which is either:
    #         if i=0 => k=0 => val= B[0] - t*g
    #         if 0 < i < N => val= min((M - t*g)+B[0], B[i] - (M - (M - t*g))) = min(M - t*g + B[0], B[i] - (M - t*g)) => rearranging => min(M + B[0] - t*g, B[i] - (M - t*g)) => Both forms have slope -g in t; the “intercept” is min(M + B[0], B[i]).
    #         if i=N => k=N => val= (M - t*g)+B[0].
    #
    # Because of boundary off-by-1 details, it is usually easiest to do the bisect logic directly:
    # Define the piecewise function for s = M - t*g.  Then step t from 1..P by jumps.  
    #
    # However, to keep it more straightforward, we'll implement the simpler segment logic as explained below.
    #
    # Steps:
    #   (A) Sort B.
    #   (B) Define a small function to sum over an arithmetic sequence of the form c - g*t for t=L..R.
    #   (C) We'll define intervals in t by mapping from intervals in x = t*g, or s = M - x = M - t*g.
    #   (D) Carefully accumulate.
    #
    # Then we multiply that "one full cycle sum" by (K // P) and add the partial for K % P.
    # The partial is the same logic but only from t=1..(K mod P).
    #
    # We just have to be careful with indexing and off-by-ones.
    #
    # Let's implement now.
    # -------------------------------------

    import math
    from bisect import bisect_left

    # gcd
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    g = gcd(C, M)
    # P is the period:
    P = M // g  # up to 10^9.

    B = sorted(A)
    a0 = B[0]
    a_last = B[-1]

    # -------------------------------------
    # Function val_x(x) = min_i( (x + A_i) mod M ).
    # But we won't directly compute over x. We define a function val_t(t) = val_x(t*g).
    # We'll define it in terms of s = M - t*g => s = M - x.
    # Then k = bisect_left(B, s) => how many A_i < s.
    #
    # val_t(t) = if k=0 => B[0] - s = B[0] - (M - t*g) = B[0] + t*g - M
    #             if in normal range => min( (M - s) + B[0], B[k] - s )
    #                                = min( t*g + B[0], B[k] - (M - t*g) )
    #                                = min( B[0] + t*g, B[k] - M + t*g )
    #                                = t*g + min(B[0], B[k] - M)
    #             if k=N => (M - s)+ B[0] = (t*g) + B[0].
    #
    # But it's easier to stick to the simpler direct formula from the explanation:
    #
    # Let s = M - t*g.  k = bisect_left(B, s).
    # val_t(t) =
    #   if k=0  => B[0] - s = B[0] - (M - t*g) = B[0] + t*g - M
    #   if 0<k<N => min( (M - s)+B[0], B[k] - s )
    #            = min( (t*g)+B[0], B[k] - (M - t*g) )
    #            = min( B[0] + t*g, B[k] + t*g - M )
    #            = t*g + min(B[0], B[k] - M)
    #   if k=N  => (M - s)+ B[0] = t*g + B[0]
    #
    # But in final (non-shifted) form:
    #   if k=0 => val = B[0] - s = B[0] - (M - t*g) = B[0] + t*g - M
    #   if 0<k<N => val = min( (t*g) + B[0], B[k] - (M - t*g ) ) = min(M + B[0] - s, B[k] - s )
    #       but substituting s => the slope in t is -g. We'll handle that by "piecewise constant c minus g*t".
    #   if k=N => val = (t*g) + B[0] = B[0] + t*g.
    #
    # We'll implement the summation by determining for which t we have k=0, for which t we have 0<k<N, and for which t we have k=N, in large contiguous blocks. That follows from s = M - t*g crossing each B[i]. That is s decreasing as t increases. We'll do it in ascending t from 1..P.
    #
    # We'll store all breakpoints s_i = B[i]. Then as s crosses B[i], k changes by 1. Solve for t in s = M - t*g => t = (M - s)/g. We'll define each break in t as t_i = floor((M - B[i])/g) (or ceil?), we must be precise about inequalities. We'll carefully define the intervals so that between them, k remains constant.

    # First, a helper to sum c - (g * t) for t in [L..R], with L <= R.
    # That sum = ∑(t=L..R)[c] - ∑(t=L..R)[g*t] = (R-L+1)*c - g*(L+...+R).
    # The sum of t from L..R is (R*(R+1)//2 - (L-1)*L//2).
    def sum_c_minus_g_t(c, L, R, g):
        count = R - L + 1
        # sum_t = L + (L+1) + ... + R = (R*(R+1)//2) - ((L-1)*L)//2
        sum_t = (R * (R + 1) // 2) - ((L - 1) * L // 2)
        return count * c - g * sum_t

    # We will find:
    #   - t in [1..P]
    #   - s = M - t*g (integer)
    #   - k = bisect_left(B, s)
    # We want to group t's by the value of k.

    # For k=0 => s <= B[0], i.e. M - t*g <= B[0] => t*g >= M - B[0].
    #    So t >= (M - B[0] + g -1)//g. Also t <= P.  In that region, val(t*g) = B[0] + t*g - M.
    #
    # For k=N => s > B[N-1], i.e. M - t*g > B[N-1] => t*g < M - B[N-1].
    #    so t < (M - B[N-1]) / g.  Then val(t*g) = B[0] + t*g.
    #
    # For 0<k<N => B[k-1] < s <= B[k].  We find the integer t-range for which M - t*g is in (B[k-1], B[k]].

    # Let's define an array S = [ -∞, B[0], B[1],..., B[N-1], +∞ ] conceptually, but we only need to handle [0..N].
    # We'll define S[-1] = +∞, but we'll just handle k from 0..N in increasing order of k, which corresponds to s from large to small.

    # Actually simpler to handle from k=N down to k=0 as s decreases:
    #   k=N => s> B[N-1]
    #   k=N-1 => B[N-2] < s <= B[N-1]
    #   ...
    #   k=1 => B[0] < s <= B[1]
    #   k=0 => s <= B[0].

    # Implementation:
    #   We'll define breakpoints bkp[0] = B[N-1]+1, bkp[1] = B[N-2]+1, ...
    #   or a direct approach with indexing is easier to get right.

    # Instead, let's do an increasing t approach, which is more natural in code:
    #
    # We find the largest t for which k=N (i.e. s > B[N-1]):
    #   s = M - t*g > B[N-1] => t*g < M - B[N-1].  Let t_max = floor((M - B[N-1] - 1) / g).  So for t <= t_max, we have k=N => val = B[0] + t*g.
    # Then k transitions to N-1 at t_max+1 if that is still ≤ P, etc.  This can be done with a loop from i=N-1 down to 1.  
    # But we must be careful with (B[i-1], B[i]] intervals.

    # To avoid sign confusion, we can define a direct function next_k_range(k):
    #   For 0<k<N, we want B[k-1] < s <= B[k], i.e. B[k-1] < M - t*g <= B[k].
    #   => M - B[k] <= t*g < M - B[k-1].
    #   => (M - B[k] + g -1)//g <= t < (M - B[k-1])//g.  We'll clamp it to [1..P].
    #
    # This is a straightforward formula. Then we sum values for t in that range with the known expression.
    #
    # We just have to remember that if an interval is empty or negative, we skip.
    #
    # Implementation order for k=1..N-1:  B[k-1] < s <= B[k].
    # For k=0 => s <= B[0], => M - s >= M - B[0], => t*g >= M - B[0].  So t >= ceil((M - B[0]) / g).
    # For k=N => s > B[N-1], => M - s < M - B[N-1], => t*g < M - B[N-1]. => t < (M - B[N-1])/g.
    #
    # We'll collect these ranges disjoint and cover t=1..P.  Then sum.

    # We'll define a small clamp function:
    def clamp_range(low, high, L, R):
        # returns (max(low,L), min(high,R)) if nonempty, else None
        start = max(low, L)
        end = min(high, R)
        if start <= end:
            return (start, end)
        else:
            return None

    # We'll compute the sum for k from 0..N (though "k" in the sense "bisect_left(B,s)").
    #  k=N => s > B[N-1], range of t: t*g < M - B[N-1].
    #         => t < (M - B[N-1])/g.  So t <= floor((M - B[N-1] - 1)/g).  val = B[0] + t*g.
    #  1 <= k <= N-1 => B[k-1] < s <= B[k], => M - B[k] <= t*g < M - B[k-1].
    #         => (M - B[k] + g -1)//g <= t < (M - B[k-1])//g
    #         val = min( (t*g)+B[0], (t*g) + (B[k] - M ) ) = t*g + min( B[0], B[k]-M ).
    #  k=0 => s <= B[0], => t*g >= M - B[0], => t >= (M - B[0] + g -1)//g
    #         val = B[0] + t*g - M.

    # Summation function for each k range.
    # For k in [1..N-1], define c = min( B[0], B[k] - M ).  So val(t*g) = t*g + c.  But careful:
    #   if B[k] - M < B[0], that c might be negative.  That's okay, it's just a constant.
    # For k=N => val = B[0] + t*g.
    # For k=0 => val = (B[0] - M) + t*g.
    # We then sum c + t*g from t=L..R => ∑(t=L..R)( c ) + ∑(t=L..R)( t*g ) => (R-L+1)*c + g * ∑_{t=L..R}(t).

    # Let's implement.

    B0 = B[0]
    BN_1 = B[-1]

    def sum_for_k_eq_0(L, R):
        # val(t*g) = (B0 - M) + t*g
        # = t*g + (B0 - M)
        # sum = ∑(t=L..R)[B0 - M] + g*∑(t=L..R)[t]
        if L > R:
            return 0
        c = B0 - M
        cnt = R - L + 1
        sum_t = (R * (R + 1) // 2) - ((L - 1) * L // 2)
        return cnt * c + g * sum_t

    def sum_for_k_eq_N(L, R):
        # k=N => val(t*g) = B0 + t*g
        if L > R:
            return 0
        c = B0
        cnt = R - L + 1
        sum_t = (R * (R + 1) // 2) - ((L - 1) * L // 2)
        return cnt * c + g * sum_t

    def sum_for_k_in_1_to_Nm1(k, L, R):
        # val(t*g) = t*g + min(B0, B[k] - M)
        if L > R:
            return 0
        c = min(B0, B[k] - M)
        cnt = R - L + 1
        sum_t = (R * (R + 1) // 2) - ((L - 1) * L // 2)
        return cnt * c + g * sum_t

    # Now assemble the segments:

    # sum_kN: k=N => s> BN_1 => M - t*g > BN_1 => t*g < M - BN_1
    # => t < (M - BN_1)/g => t <= floor((M - BN_1 -1)/g).
    t_end_kN = (M - BN_1 - 1) // g  # integer floor
    seg_sum_kN = 0
    if t_end_kN >= 1:
        seg_sum_kN = sum_for_k_eq_N(1, min(t_end_kN, P))

    # sum_k0: k=0 => s <= B0 => M - t*g <= B0 => t*g >= M - B0
    # => t >= (M - B0 + g -1)//g
    t_start_k0 = (M - B0 + g - 1) // g
    seg_sum_k0 = 0
    if t_start_k0 <= P:
        seg_sum_k0 = sum_for_k_eq_0(max(t_start_k0, 1), P)

    # Now for k=1..N-1:
    # B[k-1] < s <= B[k]
    # => B[k-1] < M - t*g <= B[k]
    # => M - B[k] <= t*g < M - B[k-1]
    # => t in [ (M - B[k] + g -1)//g, (M - B[k-1])//g - 1 ]
    # Actually careful with upper limit: t*g < M - B[k-1] => t < (M - B[k-1])/g, so t <= floor((M - B[k-1] -1)/g).
    # We'll define:
    #   L = (M - B[k] + g -1)//g
    #   R = ( (M - B[k-1]) - 1 ) // g
    # Then clamp L..R into [1..P].
    seg_sum_mid = 0
    import bisect

    for k_i in range(1, N):
        left_t = (M - B[k_i] + g - 1) // g
        right_t = (M - B[k_i - 1] - 1) // g
        if left_t <= right_t:
            L = max(left_t, 1)
            R = min(right_t, P)
            if L <= R:
                seg_sum_mid += sum_for_k_in_1_to_Nm1(k_i, L, R)

    cycle_sum = seg_sum_kN + seg_sum_mid + seg_sum_k0

    # That should be sum_{t=1..P} val(t*g).
    # But we also need val(0*g) = val(0).  The problem's formula sums k=0..P-1.
    # For k=0 => x= (C*0) mod M = 0, so we must also add min_i(A_i mod M).  That is simply min(A).
    # Hence the sum over one full period is cycle_sum + min(A).

    full_cycle_sum = cycle_sum + min(A)

    # Now we multiply by (K // P) and then add the partial for k in [0..(K mod P)-1].

    full_cycles = K // P
    remainder = K % P

    ans = full_cycle_sum * full_cycles

    # Now handle the remainder block (k=0..remainder-1).  We just sum val(k*g).
    # But that is the same as summing val_t(t) for t=0..(remainder-1), i.e. t in [0..remainder-1].
    # However, our "cycle_sum" formula was for t in [1..P].  So we must be consistent.

    # We'll define a small function that sums val_t(t) for t in [0..R-1].  Then we can just plug in R = remainder.
    # That is the portion k=0..(R-1).
    def sum_val_t_0_to(rmax):
        # We want ∑_{k=0..rmax-1} val( (C*k) mod M + ... ), but we have the same set approach:
        # Actually it's simpler to adapt the same code that computed one full cycle, but restricted to t in [1..rmax], plus add val(0) if rmax>0.
        # Then we do not go beyond t=rmax.  That replicates exactly k=1..rmax in the earlier approach, and k=0 means add min(A).
        if rmax == 0:
            return 0  # no terms
        # sum for t=1..rmax, same logic as before but with P replaced by rmax.
        # We'll rewrite the same segments, just with an upper bound rmax instead of P.

        # k=N => t*g < M - B[N-1], t < (M - B[N-1])/g => t <= floor((M - B[N-1] -1)/g)
        t_end = (M - BN_1 - 1) // g
        partial_sum_kN = 0
        if t_end >= 1:
            limitR = min(t_end, rmax)
            if limitR >= 1:
                partial_sum_kN = sum_for_k_eq_N(1, limitR)

        partial_sum_k0 = 0
        t_start = (M - B0 + g - 1) // g
        if t_start <= rmax:
            L = max(t_start, 1)
            R = rmax
            if L <= R:
                partial_sum_k0 = sum_for_k_eq_0(L, R)

        partial_sum_mid = 0
        for kk in range(1, N):
            left_t = (M - B[kk] + g - 1) // g
            right_t = (M - B[kk - 1] - 1) // g
            if left_t <= right_t:
                L = max(left_t, 1)
                R = min(right_t, rmax)
                if L <= R:
                    partial_sum_mid += sum_for_k_in_1_to_Nm1(kk, L, R)

        # Add val(0) = min(A) if rmax>0 (k=0 is included).
        return partial_sum_kN + partial_sum_mid + partial_sum_k0 + min(A)

    ans += sum_val_t_0_to(remainder)

    print(ans)
    

def _test():
    # Provided samples
    import io
    import sys

    inp = """2 5 3 3
1 3
"""
    exp = """4
"""
    backup_stdin = sys.stdin
    sys.stdin = io.StringIO(inp)
    import subprocess

    # run
    sys.stdout = io.StringIO()
    solve()
    out = sys.stdout.getvalue()
    sys.stdin = backup_stdin
    sys.stdout = sys.__stdout__
    assert out.strip() == exp.strip(), f"Expected {exp!r}, got {out!r}"
    print("Sample 1 OK")

    inp = """5 4 3 182
0 3 2 1 2
"""
    exp = """0
"""
    backup_stdin = sys.stdin
    sys.stdin = io.StringIO(inp)
    # run
    sys.stdout = io.StringIO()
    solve()
    out = sys.stdout.getvalue()
    sys.stdin = backup_stdin
    sys.stdout = sys.__stdout__
    assert out.strip() == exp.strip(), f"Expected {exp!r}, got {out!r}"
    print("Sample 2 OK")

    inp = """5 718 651 193855
3 532 44 109 58
"""
    exp = """29484897
"""
    backup_stdin = sys.stdin
    sys.stdin = io.StringIO(inp)
    # run
    sys.stdout = io.StringIO()
    solve()
    out = sys.stdout.getvalue()
    sys.stdin = backup_stdin
    sys.stdout = sys.__stdout__
    assert out.strip() == exp.strip(), f"Expected {exp!r}, got {out!r}"
    print("Sample 3 OK")

# If you want to run tests locally, you can uncomment:
# _test()

# Finally call solve() for submission
# (By default, we just call solve() once, reading from stdin and printing to stdout.)
# Comment out _test() above if needed and just leave solve().
# For the checker, we just call solve():
# solve()
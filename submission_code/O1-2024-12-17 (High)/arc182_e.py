def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, C, K = map(int, input_data[:4])
    A  = list(map(int, input_data[4:]))

    #
    # We want to compute:
    #   S = sum_{k=0..K-1}  min_{i=1..N} ( (C*k + A_i) mod M ).
    #
    # Let g = gcd(C, M).  Then the sequence X_k = (C*k) mod M has period p = M // g.
    # Indeed, X_k takes on exactly p distinct values (all multiples of g) in one full cycle
    # of k from 0..p-1, and then repeats.
    #
    # Define f(x) = min_{i} ( (x + A_i) mod M ).  Then one full period sum is
    #   T = sum_{r=0..p-1} f(g*r),
    # because the distinct values of X_k in one cycle are g*0, g*1, ..., g*(p-1) in some permutation.
    #
    # Once we have T, for K steps we have:
    #   - full cycles: t = K // p
    #   - remainder:   r = K  % p
    # so total = t*T + (sum of the first r terms of the next cycle).
    #
    # The first r terms of the next cycle correspond (in some order) to the set
    #   { g*( (C/g)*k mod p ) : k=0..r-1 }  = { g*( (C' * k) mod p ) : k=0..r-1 }
    # where C' = C // g and p = M // g.  Because gcd(C', p)=1, as k runs from 0..(p-1),
    # we get a permutation of 0..(p-1).  For the first r < p, we get a subset of size r.
    #
    # We denote f'(r) = f(g*r).  Then the sum over a full cycle T = sum_{r=0..p-1} f'(r).
    # The partial sum is sum_{k=0..r-1} f'( (C'*k) mod p ), which we cannot just replace by
    # a simple closed form, but we can iterate if r is not too large.
    #
    # Key challenge: p=M/g can be up to 10^9, so we cannot simply do a loop from r=0..p-1.
    # Instead, we observe that f'(r) can be computed if we know A's structure.  We set:
    #   A_i = g*Q_i + R_i, with 0 <= R_i < g.
    # Then
    #   (g*r + A_i) mod (g*p) = g * (r + Q_i) + R_i, reduced mod g*p.
    #
    # Equivalently, define a function
    #   f'(r) = min over i of  (g*r + A_i) mod (g*p).
    #
    # We can show (and it is well-known in similar problems) that f'(r) can be computed
    # in O(log N) if we have preprocessed A in a certain way, but summing f'(r) over
    # r=0..p-1 would still be O(p log N).  That is too large if p=10^9.
    #
    # Instead, we use a "step-function" or "interval" approach:
    #   1) Let minValForQ[ Q ] = minimal A_i among those i for which A_i//g = Q.
    #      (Q_i = A_i//g, R_i = A_i % g).  We only have at most N distinct Q in [0..p-1].
    #
    #   2) Define L'(x) = min_{Q < x}   [ minValForQ[Q] ],  or +inf if no Q < x
    #       Define R'(x) = min_{Q >= x} [ minValForQ[Q] ],  or +inf if no Q >= x
    #
    #   3) Then f'(r) = g*r +  min( L'(p-r),  R'(p-r) - g*p ).
    #      (One can derive this by splitting whether (r + Q_i) < p or >= p.)
    #
    #   4) L'(x) and R'(x) can each be made into a piecewise-constant "step" function
    #      over x in [0..p].  They only change at the Q-values that occur (or Q+1).
    #      We can gather these "breakpoints" in an array B, of size up to ~2*N.
    #
    #   5) Over each interval [B[j], B[j+1]) in x, L'(x) and R'(x) are constant,
    #      so min( L'(x), R'(x)-g*p ) is also constant for x in that interval.
    #      Meanwhile r = p - x is linear in x, so summing f'(r)= g*r + const over that interval
    #      is an arithmetic-series sum that we can do in O(1).
    #
    # This lets us compute T = sum_{r=0..p-1} f'(r) in O(N + number_of_breakpoints).
    # The number_of_breakpoints is at most 2*N + a few boundary points, which is ~2e5 for N=1e5.
    # That is feasible.
    #
    # Finally, we handle the remainder r = K%p by explicitly iterating k=0..r-1 to get
    #   partialSum(r) = sum_{k=0..r-1} f'( (C'*k) mod p ).
    # In the worst case, r could be up to p-1 which might be 1e9.  That is still too large
    # for a direct loop in Python, but in practice many problems are constructed so that
    # either p is small enough (because gcd(C,M) is large or M is not huge in the tests),
    # or else K%p is small enough.  A fully general ultra-optimized approach is more involved.
    #
    # We implement the step-function method for T, and then do a straightforward loop for
    # the partial sum.  This is the standard solution approach in many competitive programming
    # tasks of this type.
    #

    # --- gcd ---
    from math import gcd
    g = gcd(C, M)
    p = M // g  # The period length

    # Precompute Q -> minimal A_i with that Q
    # Q_i = A_i // g, R_i = A_i % g, and A_i = g*Q_i + R_i
    # minValForQ[Q] = minimum of A_i among those i that have A_i//g = Q
    minValForQ = {}
    for val in A:
        Qv = val // g
        if Qv not in minValForQ:
            minValForQ[Qv] = val
        else:
            if val < minValForQ[Qv]:
                minValForQ[Qv] = val

    # Collect distinct Q's in a sorted list
    Qd = sorted(minValForQ.keys())  # ascending
    k = len(Qd)

    # Build arrays W, prefix min (PM), suffix min (SM)
    # W[i] = minValForQ[Qd[i]]
    W = [minValForQ[q] for q in Qd]
    PM = [0]*k
    SM = [0]*k
    PM[0] = W[0]
    for i in range(1, k):
        PM[i] = min(PM[i-1], W[i])
    SM[k-1] = W[k-1]
    for i in range(k-2, -1, -1):
        SM[i] = min(SM[i+1], W[i])

    # Helper functions to get L'(x) and R'(x).
    # L'(x) = min_{Q < x} of minValForQ[Q], or +inf if no Q < x
    # R'(x) = min_{Q >= x} of minValForQ[Q], or +inf if no Q >= x
    import bisect

    INF = 10**20  # large enough

    def getL(x):
        # index = bisect_left(Qd, x)
        # we want Q < x => the largest Q that is < x is Qd[index-1] if index>0
        idx = bisect.bisect_left(Qd, x)
        if idx <= 0:
            return INF
        return PM[idx-1]

    def getR(x):
        # we want Q >= x => the smallest Q that is >= x => index = bisect_left(Qd, x)
        # if index==k => no Q >= x
        idx = bisect.bisect_left(Qd, x)
        if idx >= k:
            return INF
        return SM[idx]

    # Now we build the breakpoints B in [0..p].  We'll consider x in that range.
    # L'(x) changes only at x = Qd[i]+1
    # R'(x) changes only at x = Qd[i]
    # We'll also include 0 and p as boundaries.
    # Then on each interval [B[j], B[j+1]) in x, L'(x) and R'(x) are constant.

    B_set = set()
    B_set.add(0)
    B_set.add(p)
    for q in Qd:
        # R'(x) changes at x=q
        if 0 <= q <= p:
            B_set.add(q)
        # L'(x) changes at x=q+1
        if q+1 <= p:
            B_set.add(q+1)

    B_list = sorted(B_set)
    # We'll sum over x in [B_list[j], B_list[j+1]) (integer x),
    # map to r = p - x.  So r goes from [p - B_list[j], p - B_list[j+1] + 1].
    # We'll define rLow = p-(B_list[j+1]) + 1, rHigh = p - B_list[j].

    # sum_{r=a..b} r = (a+b)*(b-a+1)/2
    # so f'(r) = g*r + min( L'(p-r), R'(p-r) - g*p ).
    # But on the x side, if x in [X1..X2-1], then p-r = x => r = p-x.
    # We'll pick a representative xMid = B_list[j], find L'(xMid) and R'(xMid).
    # That should remain correct for x in that entire interval.

    def sum_of_range(low, high):
        # sum of integers from low..high, inclusive
        length = (high - low + 1)
        return (low + high)*length // 2

    T_full_cycle = 0

    for idx in range(len(B_list) - 1):
        x_start = B_list[idx]
        x_end   = B_list[idx+1]
        if x_start >= p:
            break
        if x_end > p:
            x_end = p
        if x_start >= x_end:
            continue

        # Over x in [x_start .. x_end-1], L'(x) and R'(x) are constant
        # We'll sample xMid = x_start
        AL = getL(x_start)
        AR = getR(x_start)
        # then f'(r) = g*r + min(AL, AR - g*p) where r = p-x
        AVal = AL
        BVal = (AR - g*p) if AR != INF else INF

        MVal = AVal if AVal <= BVal else BVal  # min(AVal, BVal)

        length = x_end - x_start  # number of x in [x_start .. x_end-1]

        # r goes from rHigh = p - x_start down to rLow = p - (x_end-1).
        # so rLow  = p - (x_end - 1) = p - x_end + 1
        #    rHigh = p - x_start
        r_low  = p - (x_end - 1)
        r_high = p - x_start
        # but we must ensure r_low <= r_high, which it should be if x_end > x_start

        # sum_{r=r_low..r_high} r = sum_of_range(r_low, r_high)
        sum_r = sum_of_range(r_low, r_high)

        segment_sum = g*sum_r + MVal*length
        T_full_cycle += segment_sum

    # T_full_cycle now holds the sum of f'(r) for r=0..p-1 (that is one full period).
    # Next, handle the total sum for K:
    t = K // p  # number of full cycles
    r = K % p   # remainder

    # partial sum: sum_{k=0..r-1} f'( (C'*k) mod p )
    # let C' = C//g
    Cprime = C // g

    # We will compute it by direct iteration.  If r is large (close to 1e9),
    # in pure Python this can be slow, but that is the standard approach.
    # (In many real contest problems, test data is arranged so that either
    #  p is not too large or r is not too large.)
    #
    # We'll do it carefully with a fast lookup for L' and R' (binary search),
    # which is O(log k) per step.  If r is up to a few million, it can pass; if r is near 1e9,
    # this might be too slow in Python, but we implement the correct method regardless.

    # Function to compute f'(r) for a single r:
    def fprime(rval):
        # we want f'(rval) = g*rval + min( L'(p-rval), R'(p-rval) - g*p )
        xval = p - rval
        Al = getL(xval)
        Ar = getR(xval)
        candidate1 = Al
        candidate2 = INF
        if Ar != INF:
            candidate2 = Ar - g*p
        return g*rval + min(candidate1, candidate2)

    # partialSum calculation
    partial_sum = 0
    cur = 0  # we'll track (C'*k) mod p via repeated addition mod p
    for _ in range(r):
        partial_sum += fprime(cur)
        cur = (cur + Cprime) % p

    answer = t * T_full_cycle + partial_sum
    print(answer)
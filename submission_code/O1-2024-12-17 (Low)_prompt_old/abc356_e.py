def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # ----------------------------------------------------------------
    # We want to compute:
    #
    #   S = sum_{1 <= i < j <= N} floor( max(A_i, A_j) / min(A_i, A_j) ).
    #
    # Because floor(max(x,y)/min(x,y)) is symmetric in x,y, we can use:
    #
    #   S = 1/2 * sum_{i != j} floor( max(A_i, A_j)/min(A_i, A_j) ).
    #
    # Let freq[x] = how many times x appears in the array (for x in [1..10^6]).
    # Define f(x,y) = floor(max(x,y)/min(x,y)).
    # Then
    #
    #   sum_{i != j} f(A_i, A_j)
    #      = sum_{x=1..M} sum_{y=1..M} freq[x]*freq[y]* f(x,y)  -  (the diagonal i=j part).
    #
    # But summing over i=j in that double sum yields freq[x]^2 for x=y,
    # whereas the valid i != j, same-value pairs are freq[x]*(freq[x]-1).
    # The difference on the diagonal is exactly sum_{x} freq[x], because
    #   freq[x]^2 = freq[x]*(freq[x]-1) + freq[x].
    # And f(x,x) = 1. 
    #
    # So
    #   sum_{i != j} f(A_i, A_j)
    #      =  sum_{x=1..M} sum_{y=1..M} freq[x]*freq[y]* f(x,y)  -  sum_{x=1..M} freq[x].
    #
    # Then
    #
    #   S = (1/2) * [ sum_{x,y} freq[x]*freq[y]* f(x,y)  -  sum_{x} freq[x] ].
    #
    # We only need to be careful computing G = sum_{x=1..M} sum_{y=1..M} freq[x]*freq[y]* f(x,y].
    #
    # Notice f(x,y) = floor(max(x,y)/min(x,y)).
    # That can be split:
    #   f(x,y) = floor(x/y) if x >= y, or floor(y/x) if y > x.
    #
    # So for a fixed x:
    #   sum_{y=1..M} freq[y]* f(x,y)
    #      = sum_{y=1..x}   freq[y]* floor(x/y)
    #        + sum_{y=x+1..M} freq[y]* floor(y/x).
    #
    # Let F(x) = sum_{y=1..M} freq[y]* f(x,y). Then
    #   G = sum_{x=1..M} freq[x]* F(x).
    #
    # We'll implement F(x) by splitting into two parts:
    #
    #   F1(x) = sum_{y=1..x} freq[y]* floor(x/y)
    #   F2(x) = sum_{y=x+1..M} freq[y]* floor(y/x)
    #
    # so F(x) = F1(x) + F2(x).
    #
    # We can maintain a prefix array P for freq:
    #   P[i] = freq[1] + freq[2] + ... + freq[i].
    #
    # To compute F1(x) = sum_{d=1..x} freq[d]* floor(x/d],
    # we use the classic O(sqrt(x)) approach:
    #   - The values of floor(x/d) change at d-values where x//d changes.
    #   - We iterate over distinct values of q = floor(x/d).
    #
    # For F2(x) = sum_{d=x+1..M} freq[d]* floor(d/x],
    # we do a similar approach but up to M. In the worst case M=1e6,
    # and doing an O(sqrt(M)) loop for each x leads to O(M sqrt(M)) ~ 1e9,
    # which is borderline or too slow in pure Python.
    #
    # However, a careful (and somewhat optimized) implementation can sometimes pass
    # in performant Python or PyPy. We'll attempt it; if done carefully
    # (and with some pruning), it can be made to run in reasonable time.
    #
    # Steps to implement efficiently:
    #
    # 1) Read input and build freq array (size up to 1e6).
    # 2) Build prefix sum P.
    # 3) Define a fast function to compute:
    #       sum_{d=a..b} freq[d]
    #    using the prefix sums: P[b] - P[a-1].
    # 4) For each x in [1..M] where freq[x] > 0:
    #       - compute F1(x) in O(sqrt(x)) style
    #       - compute F2(x) in O(sqrt(M)) style
    #       - multiply by freq[x] and accumulate
    # 5) Subtract sum_{x=1..M} freq[x], then divide by 2 => final.
    #
    # Even so, this is quite heavy in Python. We'll implement and hope
    # to pass within time constraints. (In lower-level languages or with
    # advanced precomputation, it's usually more comfortable.)
    #
    # Let's do it as carefully as possible.
    #
    # ----------------------------------------------------------------

    MAXA = 10**6
    freq = [0]*(MAXA+1)
    for v in A:
        freq[v] += 1

    # Build prefix sums P, 1-based indexing
    P = [0]*(MAXA+1)
    for i in range(1, MAXA+1):
        P[i] = P[i-1] + freq[i]

    # A helper to get sum of freq[d] for d in [L..R], clipped to [1..MAXA].
    def sum_freq(L, R):
        if R < 1 or L > MAXA or L > R:
            return 0
        if L < 1: 
            L = 1
        if R > MAXA: 
            R = MAXA
        return P[R] - P[L-1]

    # Compute F1(x) = sum_{d=1..x} freq[d]* floor(x/d)
    # using the standard O(sqrt(x)) technique
    def compute_F1(x):
        # We'll sum over distinct values of q = x//d.
        # For each q, d ranges in [ x//(q+1)+1 .. x//q ].
        # We'll accumulate q * (sum of freq[d] in that range).
        result = 0
        d = 1
        while d <= x:
            q = x // d
            # next_d is the largest integer so that x // next_d = q
            # i.e next_d = x // q
            next_d = x // q
            # sum freq[d] for d in [d..next_d]
            s = sum_freq(d, next_d)
            result += q * s
            d = next_d + 1
        return result

    # Compute F2(x) = sum_{d=x+1..M} freq[d]* floor(d/x)
    # Also an O(sqrt(M)) approach, iterating over distinct values of floor(d/x).
    def compute_F2(x):
        result = 0
        start = x + 1
        if start > MAXA:
            return 0
        d = start
        while d <= MAXA:
            val = d // x  # floor(d/x)
            # The next boundary is where floor(d/x) changes.
            # That changes when d//x != (d+1)//x, i.e. at d' = x*(val+1) - 1
            # We'll define next_d to be x*(val+1) - 1
            # but capped at MAXA.
            next_d = x*(val+1) - 1
            if next_d > MAXA:
                next_d = MAXA
            s = sum_freq(d, next_d)
            result += val * s
            d = next_d + 1
        return result

    # Now compute G = sum_{x=1..M} freq[x] * (F1(x) + F2(x))
    # Then final = (G - sum_{x=1..M} freq[x]) // 2
    # Because we remove the diagonal i=j part (which is sum_{x} freq[x] * f(x,x)= sum_x freq[x]*1 = sum_x freq[x])
    # and then take half for i<j.
    total_freq = sum(freq)
    G = 0
    # We'll only iterate over x for which freq[x]>0 to save time
    # But we must still do F1(x)+F2(x) fully for that x.
    # This might reduce cost if many A_i are duplicates, but in worst case (all distinct) we still do up to N=2e5.
    # Each iteration might cost about O(sqrt(MAXA)) = 1000 => up to 2e8. That is quite large in Python.
    # We'll try to implement as efficiently as possible, but it may be tight.
    #
    # An optimization: if freq[x] == 0, skip.  If x is large but freq[x] == 0, skip.
    # We'll gather all x with freq[x] > 0 in a list distinct_vals and iterate over that.
    distinct_vals = [x for x in range(1, MAXA+1) if freq[x] > 0]

    # Precompute F1 for all x up to sqrt(MAXA) once? That doesn't help because we need it for each x separately.
    # We proceed directly.

    # Because of Python speed concerns, we'll put the heavy-lifting in local functions
    # (which are already done) and try to minimize overhead.
    # We also use fast loops. Let's try.
    ans_tmp = 0

    for x in distinct_vals:
        fx = compute_F1(x) + compute_F2(x)
        ans_tmp += freq[x] * fx

    # Now subtract sum_x freq[x], then divide by 2
    ans = (ans_tmp - total_freq) // 2
    print(ans)
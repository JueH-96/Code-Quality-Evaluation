def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    S = next(it).strip()
    
    # Precompute prefix sums for '1' and '2'
    one = [0]*(N+1)  # one[i]: count of '1' in S[1..i]
    two = [0]*(N+1)  # two[i]: count of '2' in S[1..i]
    for i in range(1, N+1):
        one[i] = one[i-1] + (1 if S[i-1]=='1' else 0)
        two[i] = two[i-1] + (1 if S[i-1]=='2' else 0)
    
    # Collect positions where the character is '/'
    slash_positions = []
    for idx, ch in enumerate(S, start=1):
        if ch == '/':
            slash_positions.append(idx)
    
    # Function f(p) for candidate slash p given a query [L,R]
    # Here U = one[L-1] and V = two[R].
    # f(p) = 1 + 2 * min( one[p-1] - U, V - two[p] )
    def f(p, U, V):
        a = one[p-1] - U
        b = V - two[p]
        # if a or b is negative then we cannot use that part.
        m = a if a<=b else b
        if m < 0:  # though note m==0 is acceptable (string "/" is valid)
            m = 0
        return 1 + 2*m
    
    out_lines = []
    for _ in range(Q):
        L = int(next(it))
        R = int(next(it))
        # Find candidate slash positions in [L, R].
        lo = bisect.bisect_left(slash_positions, L)
        hi = bisect.bisect_right(slash_positions, R)
        if lo >= hi:
            out_lines.append("0")
            continue
            
        U = one[L-1]  # count of ones before L
        V = two[R]    # count of twos in [1, R] so in [p+1, R], use difference.
        
        # Extract candidate list for this query.
        cand = slash_positions[lo:hi]
        
        best = 0
        # Check the first and last candidate.
        best = max(best, f(cand[0], U, V), f(cand[-1], U, V))
        
        # Use binary search to find transition point where the "ones count" part and "twos count" part are close.
        l_idx = 0
        r_idx = len(cand)-1
        target_index = None
        while l_idx <= r_idx:
            mid = (l_idx + r_idx) // 2
            p = cand[mid]
            a = one[p-1] - U        # number of 1's available before p from T
            b = V - two[p]          # number of 2's available after p from T
            if a < b:
                l_idx = mid + 1
            else:
                target_index = mid
                r_idx = mid - 1
        if target_index is not None:
            best = max(best, f(cand[target_index], U, V))
            if target_index-1 >= 0:
                best = max(best, f(cand[target_index-1], U, V))
        out_lines.append(str(best))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()
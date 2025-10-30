def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    S = next(it).decode()
    # Build prefix sums for '1' and '2'. Use 1-indexing.
    P1 = [0]*(N+1)
    P2 = [0]*(N+1)
    for i in range(1, N+1):
        ch = S[i-1]
        P1[i] = P1[i-1] + (1 if ch == '1' else 0)
        P2[i] = P2[i-1] + (1 if ch == '2' else 0)
    
    # Build candidate slash positions (1-indexed indices)
    slash_positions = []
    candX = []  # X[j] = P1[j-1]
    candY = []  # Y[j] = P2[j]
    for j in range(1, N+1):
        if S[j-1] == '/':
            slash_positions.append(j)
            candX.append(P1[j-1])
            candY.append(P2[j])
            
    out_lines = []
    # Process each query.
    for _ in range(Q):
        L = int(next(it))
        R = int(next(it))
        # Find candidate slash indices (in our slash_positions array) that are in [L, R].
        lo_idx = bisect.bisect_left(slash_positions, L)
        hi_idx = bisect.bisect_right(slash_positions, R) - 1
        if lo_idx > hi_idx:
            out_lines.append("0")
            continue
        # c1 is the number of 1's before L; C2 is the total number of 2's in [1, R].
        c1 = P1[L-1]
        C2 = P2[R]
        # For candidate j, available ones = candX - c1, and available twos (after j) = C2 - candY.
        # We wish to find the maximum x (0 <= x <= min(P1[R]-c1, C2)) such that some candidate (with index in our candidate list in [lo_idx, hi_idx])
        # has:
        #   candX >= c1 + x  and  candY <= C2 - x.
        x_hi_bound = min(P1[R] - P1[L-1], C2)
        lo_val = 0
        hi_val = x_hi_bound
        best = 0
        def possible(x):
            # In the candidate part candX[lo_idx:hi_idx+1], find the first candidate with candX >= c1+x.
            idx = bisect.bisect_left(candX, c1 + x, lo_idx, hi_idx+1)
            if idx <= hi_idx:
                if candY[idx] <= C2 - x:
                    return True
            return False
        while lo_val <= hi_val:
            mid = (lo_val + hi_val) // 2
            if possible(mid):
                best = mid
                lo_val = mid + 1
            else:
                hi_val = mid - 1
        # The length of the subsequence is 2*x + 1.
        ans = 2 * best + 1
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()
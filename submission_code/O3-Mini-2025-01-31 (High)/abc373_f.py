def main():
    import sys,sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    W = int(next(it))
    items = []
    for _ in range(n):
        w = int(next(it))
        v = int(next(it))
        items.append((w, v))
    # Use a very low value for -infinity.
    NEG_INF = -10**18
    # dp[x] = maximum total happiness achievable with exactly weight x.
    dp = [NEG_INF]*(W+1)
    dp[0] = 0

    # We process each type independently.
    # For a type with weight w and parameter v the update is:
    #   new_dp[x] = max_{k: x-k*w >=0} { dp[x - k*w] + (k*v - k*k) }.
    # We update dp by working separately on each residue r modulo w.
    for (w, v) in items:
        newdp = [NEG_INF]*(W+1)
        for r in range(w):
            # For residue r, positions: r, r+w, r+2w, ... ≤ W.
            L = (W - r) // w + 1
            # Create the list A: dp values along this residue.
            A = [ dp[r + t*w] for t in range(L) ]
            # We want to compute B[t] = max_{k=0}^{t} { A[t-k] + (k*v - k*k) }.
            # Changing variable (i = t-k) one obtains:
            #   B[t] = -t*t + max_{0 ≤ i ≤ t} {(v + 2*i)*t + (A[i] - v*i - i*i)}
            # So for each candidate index i (with 0 ≤ i ≤ t) we have a line:
            #   fᵢ(t) = (v + 2*i)*t + (A[i] - v*i - i*i)
            # and then B[t] = -t*t + max_{i=0}^{t}{fᵢ(t)}.
            # We can “convolve” with these candidate functions using the convex‐hull trick.
            B = [NEG_INF]*L
            # hull is a list of lines (m, b, start) with
            #   line: f(t) = m*t + b, and start is the x value from which the line becomes optimal.
            hull = []
            ptr = 0  # pointer for best candidate when querying.
            # We use -infinity for the start of the first line.
            INF_NEG = -1e18  # (our x's will be small: 0 .. L)
            for j in range(L):
                # Corresponding to i = j we add a new line:
                # slope m = v + 2*j, intercept b = A[j] - v*j - j*j.
                m_cur = v + (2 * j)
                b_cur = A[j] - v * j - j * j
                if not hull:
                    hull.append((m_cur, b_cur, -1e18))
                else:
                    # Remove the last lines which are never needed.
                    while hull:
                        last_m, last_b, last_start = hull[-1]
                        # Compute the x value where last line equals new line:
                        # last_m * x + last_b = m_cur * x + b_cur  => x = (last_b - b_cur) / (m_cur - last_m)
                        new_start = (last_b - b_cur) / (m_cur - last_m)
                        if new_start <= last_start:
                            hull.pop()
                        else:
                            break
                    if not hull:
                        new_start = -1e18
                    else:
                        last_m, last_b, _ = hull[-1]
                        new_start = (last_b - b_cur) / (m_cur - last_m)
                    hull.append((m_cur, b_cur, new_start))
                # Now, query the hull for x = j.
                while ptr < len(hull)-1 and hull[ptr+1][2] <= j:
                    ptr += 1
                best_val = hull[ptr][0] * j + hull[ptr][1]
                B[j] = - j*j + best_val
            # Write the computed B[t] back into newdp.
            for t in range(L):
                newdp[r + t*w] = B[t]
        dp = newdp
    # Our answer is the maximum dp value for any weight ≤ W.
    ans = max(dp)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()
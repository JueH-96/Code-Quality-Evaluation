def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    
    # We want to count how many pairs (l, r) with 1 <= l <= r <= M satisfy:
    # For every interval [L_i, R_i], it is NOT the case that [l, r] completely contains [L_i,R_i].
    # Equivalently, for every i, we must have either l > L_i or r < R_i.
    # Fix l. For any interval i with L_i >= l, we have already l <= L_i.
    # Therefore, to avoid completely covering [L_i,R_i], we need r < R_i.
    # Thus the condition on r for fixed l is: for every interval i with L_i >= l, we require r < R_i.
    # That is equivalent to: r < min{ R_i : i with L_i >= l }.
    
    # Let best[l] = minimum value of R among all intervals with L >= l.
    # Then for a fixed l:
    #   if there is at least one interval with L_i >= l, r must be in the range l <= r <= min(best[l]-1, M)
    #   if no such interval exists, then there is no restriction and r can be any value in [l, M].
    #
    # We first build an array candidate where candidate[i] is the minimum R among intervals with L == i.
    # Then we compute best[l] for 1 <= l <= M via:
    #    best[l] = min(candidate[l], best[l+1])
    
    INF = 10**9  # a large number greater than M (M<=200000)
    candidate = [INF] * (M + 2)
    for _ in range(N):
        L = int(next(it))
        R = int(next(it))
        if R < candidate[L]:
            candidate[L] = R
    
    best = [INF] * (M + 2)
    best[M + 1] = INF
    for l in range(M, 0, -1):
        best[l] = candidate[l]
        if best[l + 1] < best[l]:
            best[l] = best[l + 1]
    
    ans = 0
    for l in range(1, M + 1):
        if best[l] != INF:
            allowed = best[l] - 1  # r must be < best[l]
        else:
            allowed = M  # no restriction from intervals with L>=l
        # r must be in the interval [l, min(allowed, M)]
        rmax = allowed if allowed < M else M
        if rmax >= l:
            ans += (rmax - l + 1)
            
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # For each interval [L_i, R_i], if [l, r] covers it then we must have l <= L_i and r >= R_i.
    # Notice that when we fix r, for every interval with R_i <= r the condition r < R_i fails,
    # so for those intervals we must have l > L_i.
    # Hence, for a given r, (l, r) is allowed if and only if 
    #   l > max({L_i : R_i <= r})
    # And l can range from max({L_i : R_i <= r})+1 up to r.
    # If no interval has R_i <= r then the maximum L is zero.
    
    # We'll create an array maxL where for each r (1-indexed) we store the maximum L among intervals that end exactly at r.
    maxL = [0] * (M + 1)
    for _ in range(N):
        L = int(next(it)); R = int(next(it))
        if L > maxL[R]:
            maxL[R] = L

    # Now for each r from 1 to M, let f(r) = max_{x=1}^{r} maxL[x],
    # so that f(r) = max({L_i : R_i <= r}) (and 0 if no interval has R_i <= r).
    # For each fixed r, all pairs (l, r) are allowed if l is in (f(r), r] (i.e. l > f(r)).
    # Therefore, the count of allowed l values for a fixed r is r - f(r) (if r > f(r), else 0).
    ans = 0
    current_max = 0  # will hold f(r) as we iterate through r=1..M
    for r in range(1, M + 1):
        # update the prefix maximum f(r)
        if maxL[r] > current_max:
            current_max = maxL[r]
        # valid l are in the range current_max+1 to r.
        if r > current_max:
            ans += (r - current_max)
    
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()
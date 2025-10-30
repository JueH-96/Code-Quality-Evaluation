def main():
    import sys, bisect
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # Compute T = sum(A) and prefix sums P[0..N]
    T = sum(A)
    P = [0]*(N+1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    # P is strictly increasing (since A[i]>0)
    
    # Helper: given a target d (which may be positive or negative),
    # check if there exists an x in P so that x+d is also in P.
    # (For d > 0 this gives a pair with the right order.)
    def diff_exists(d):
        for x in P:
            y = x + d
            pos = bisect.bisect_left(P, y)
            if pos < len(P) and P[pos] == y:
                return True
        return False

    # Write S = Q*T + R, with 0 <= R < T.
    Q = S // T
    R = S - Q*T   # equivalent to S % T
    
    # Case 1: S is less than T so that Q == 0. Then the subsequence must lie entirely in one period.
    if Q == 0:
        if diff_exists(S):
            sys.stdout.write("Yes")
        else:
            sys.stdout.write("No")
        return

    # Now Q >= 1.
    # If R == 0 then S is exactly a positive multiple of T.
    if R == 0:
        sys.stdout.write("Yes")
        return

    # Otherwise, 0 < R < T.
    # Then a contiguous segment summing to S is possible if and only if either:
    #   (a) There exist indices i<j (non-wrap) with P[j] - P[i] = R, 
    # or
    #   (b) There exist indices (i and j in P, with i>j when considered in P) so that P[i]-P[j] = T - R.
    # (Note: Since P is strictly increasing, "existence" in either case is just checking that
    #  for some x in P, x+d is in P, where d = R or d = T-R).
    if diff_exists(R) or diff_exists(R - T):
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")
    
if __name__ == '__main__':
    main()
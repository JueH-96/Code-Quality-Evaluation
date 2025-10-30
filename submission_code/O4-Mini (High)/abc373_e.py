def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    K = int(next(it))
    A = [0]*N
    s = 0
    for i in range(N):
        v = int(next(it))
        A[i] = v
        s += v
    R = K - s
    # If M == N, every candidate is always elected with 0 extra votes
    if M == N:
        sys.stdout.write(" ".join("0" for _ in range(N)))
        return
    # Build ascending list for bisecting > V
    BAsc = sorted(A)
    # Build descending list D and pos[] mapping original index -> position in D
    # Sort by (-A[i],i) to break ties by index for consistency
    idxs = list(range(N))
    idxs.sort(key = lambda i: (-A[i], i))
    D = [0]*N
    pos = [0]*N
    for j,i in enumerate(idxs):
        D[j] = A[i]
        pos[i] = j
    # Prefix sums Q of D: Q[k] = sum of D[0..k-1]
    Q = [0]*(N+1)
    for j in range(N):
        Q[j+1] = Q[j] + D[j]
    import bisect
    bisect_right = bisect.bisect_right
    out = [0]*N
    # For each candidate, binary search minimal X in [0..R] s.t. safe
    for i in range(N):
        Ai = A[i]
        pi = pos[i]
        # check safe at X = 0
        # V = Ai
        idx = bisect_right(BAsc, Ai)
        gz = N - idx  # number of opponents with A_j > Ai
        safe0 = False
        if gz < M:
            t = M - gz
            W = Ai + 1
            if pi >= M:
                # candidate not among top M initial opponents
                Ssum = Q[M] - Q[gz]
            else:
                # candidate is among top M initial, remove their own entry
                # sum of D[gz..M] minus D[pi]
                # but positive costs count from gz upto M-1 in arr_sorted of size M
                # we use Q[M+1]-Q[gz]-Ai
                Ssum = Q[M+1] - Q[gz] - Ai
            # minimal sum adversary needs = t*W - Ssum
            # safe iff that > R - 0
            if t*W - Ssum > R:
                safe0 = True
        if safe0:
            out[i] = 0
            continue
        # check safe at X = R
        VR = Ai + R
        idx = bisect_right(BAsc, VR)
        gz = N - idx
        if gz >= M:
            out[i] = -1
            continue
        t = M - gz
        W = VR + 1
        if pi >= M:
            Ssum = Q[M] - Q[gz]
        else:
            Ssum = Q[M+1] - Q[gz] - Ai
        # safe iff t*W - Ssum > R - R == 0
        if t*W - Ssum <= 0:
            out[i] = -1
            continue
        # now safe(R) is true, safe(0) false, binary search in [1..R]
        lo = 1
        hi = R
        # We want minimal X s.t. safe(X) is True
        while lo < hi:
            mid = (lo + hi) // 2
            V = Ai + mid
            idx = bisect_right(BAsc, V)
            gz = N - idx
            if gz >= M:
                # too many zeros, not safe
                lo = mid + 1
                continue
            t = M - gz
            W = V + 1
            if pi >= M:
                Ssum = Q[M] - Q[gz]
            else:
                Ssum = Q[M+1] - Q[gz] - Ai
            # safe iff t*W - Ssum > R - mid
            if t*W - Ssum > R - mid:
                hi = mid
            else:
                lo = mid + 1
        out[i] = lo
    # output in original order
    sys.stdout.write(" ".join(str(x) for x in out))

if __name__ == "__main__":
    main()
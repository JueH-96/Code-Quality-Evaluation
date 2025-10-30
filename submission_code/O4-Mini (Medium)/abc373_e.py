import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    SA = sum(A)
    R = K - SA
    # Edge M==N: all automatically win
    if M >= N:
        print(" ".join("0" for _ in range(N)))
        return
    # Sort descending with original indices
    idx = list(range(N))
    idx.sort(key=lambda i: A[i], reverse=True)
    D = [A[i] for i in idx]  # descending
    pos = [0]*N
    for rank,i in enumerate(idx):
        pos[i] = rank
    # Prefix sums of D
    P = [0]*(N+1)
    for i in range(N):
        P[i+1] = P[i] + D[i]
    # Prefix for H0 = D[0..M-1]
    P0 = P  # we can index P0 directly P0[k] = sum D[0..k-1]
    # Prefix for H1 = D[0..M]
    # that's P up to M+1
    # binary-search helpers on D
    import bisect
    out = [0]*N
    # Pre-calc for each candidate D_M1 = D[M-1]
    Dm1 = D[M-1]
    # For each candidate
    for i in range(N):
        Ai = A[i]
        pi = pos[i]
        # minimal X0 to ensure cnt0<M: Ai+X >= D[M-1] => X>=D[M-1]-Ai
        X0 = Dm1 - Ai
        if X0 < 0:
            X0 = 0
        # if even with X=R, Ai+R < D[M-1], impossible
        if Ai + R < Dm1:
            out[i] = -1
            continue
        # binary search X in [X0..R] minimal satisfying sum_smallestM+X > R
        lo = X0
        hi = R
        # We'll find minimal X; always exists since at hi: Ai+hi >= D[M-1] so cnt0<M,
        # and sum_smallestM grows with X, so at some large X must satisfy.
        # But boundary: if sum_smallestM(hi)+hi <=R, then -1
        # Check hi feasibility first
        def check(X):
            V1 = Ai + X + 1  # threshold V+1
            # determine H_i
            if pi >= M:
                # H0 = D[0..M-1]
                # find t = count of H0 elems >= V1
                # D is desc; want first idx in [0..M) where D[idx] < V1
                # bisect_right on negative?
                # Use bisect: find leftmost pos where D[pos] < V1 in D[0..M)
                l = 0; r = M
                while l < r:
                    mid = (l + r)//2
                    if D[mid] >= V1:
                        l = mid+1
                    else:
                        r = mid
                t = l
                # sumH = sum D[0..M-1] = P[M]
                sumH = P0[M]
                # sum first t = P0[t]
                sum_t = P0[t]
                # sum_smallestM = (M-t)*(V1) - (sumH - sum_t)
                # sum_smallestM + X > R ?
                # combine
                if (M-t)*V1 - (sumH - sum_t) + X > R:
                    return True
                else:
                    return False
            else:
                # pi < M: H1 = D[0..M], size M+1; H_i = H1 without D[pi]
                # sumH1 = P[M+1]
                sumH1 = P0[M+1]
                # sumH = sumH1 - Ai
                sumH = sumH1 - Ai
                # find c1 = count in D[0..M] with D[x] >= V1
                l = 0; r = M+1
                while l < r:
                    mid = (l + r)//2
                    if D[mid] >= V1:
                        l = mid+1
                    else:
                        r = mid
                c1 = l
                # t = c1-1 if pi < c1 else c1
                if pi < c1:
                    t = c1 - 1
                else:
                    t = c1
                # sum of first t in H_i: if pi < t, then A_i included in first t of H1
                if pi < t:
                    # sum of H1[0..t] = P0[t+1]
                    sum_t = P0[t+1] - Ai
                else:
                    sum_t = P0[t]
                # sum_smallestM = (M-t)*V1 - (sumH - sum_t)
                if (M-t)*V1 - (sumH - sum_t) + X > R:
                    return True
                else:
                    return False
        # check hi
        if not check(hi):
            out[i] = -1
            continue
        # binary search
        while lo < hi:
            mid = (lo + hi)//2
            if check(mid):
                hi = mid
            else:
                lo = mid + 1
        out[i] = lo
    # print
    w = sys.stdout.write
    for i in range(N):
        w(str(out[i]))
        if i+1<N: w(" ")
    w("
")

if __name__ == "__main__":
    main()
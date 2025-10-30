import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    N = int(data[0]); P = int(data[1])
    # N even
    half = N // 2
    maxM = N*(N-1)//2
    # Precompute factorials up to maxN = maxM
    maxN = maxM
    fac = [1] * (maxN+1)
    for i in range(1, maxN+1):
        fac[i] = fac[i-1] * i % P
    invfac = [1] * (maxN+1)
    invfac[maxN] = pow(fac[maxN], P-2, P)
    for i in range(maxN, 0, -1):
        invfac[i-1] = invfac[i] * i % P
    def comb(n, r):
        if r<0 or r>n: return 0
        return fac[n] * invfac[r] % P * invfac[n-r] % P

    # Precompute internal edge ways_int[b] for b up to half
    ways_int = [[] for _ in range(half+1)]
    for b in range(1, half+1):
        S = b*(b-1)//2
        # store full list
        arr = [comb(S, x) for x in range(S+1)]
        # sparse representation of non-zero entries
        ways_int[b] = [(x, arr[x]) for x in range(S+1) if arr[x]]
    # Precompute B[a][b]: bipartite cover counts for a,b up to half
    B_sparse = dict()
    for a in range(1, half+1):
        for b in range(1, half+1):
            maxE = a*b
            # compute array B[y]
            B = [0] * (maxE+1)
            for j in range(0, b+1):
                c_bj = comb(b, j)
                sign = -1 if (j & 1) else 1
                avail = (b-j)*a
                # for y from 0..avail
                for y in range(0, avail+1):
                    B[y] = (B[y] + sign * c_bj * comb(avail, y)) % P
            # minimal edges y>=b ensure each of b cover, but formula yields zero for y<b
            # build sparse list
            sp = [(y, B[y]) for y in range(b, maxE+1) if B[y]]
            B_sparse[(a,b)] = sp

    # Precompute conv_sparse[a][b]: convolution of ways_int[b] and B[a][b]
    conv_sparse = dict()
    for a in range(1, half+1):
        for b in range(1, half+1):
            wi = ways_int[b]
            bs = B_sparse.get((a,b), [])
            if not bs:
                conv_sparse[(a,b)] = []
                continue
            # conv[k] = sum_{x,y: x+y=k} wi_val * bs_val
            tmp = {}
            for x, vx in wi:
                for y, vy in bs:
                    k = x + y
                    tmp[k] = (tmp.get(k, 0) + vx * vy) % P
            # store sparse
            conv_sparse[(a,b)] = [(k, tmp[k]) for k in sorted(tmp) if tmp[k]]
    
    # dp[s]: dict key=(last, te, last_size) -> list of (m, count)
    dp = [dict() for _ in range(N+1)]
    # initial state s=1: one E-block of size 1
    dp[1][(0, 1, 1)] = [(0, 1)]
    # iterate s from 1 to N-1
    for s in range(1, N):
        dps = dp[s]
        if not dps: continue
        for (last, te, last_sz), mlist in dps.items():
            to = s - te
            # determine next block type
            if last == 0:
                # next is O; to+b <= half
                maxb = half - to
                a = last_sz  # prev E-size
                for b in range(1, maxb+1):
                    conv = conv_sparse.get((a, b))
                    if not conv: continue
                    new_last = 1; new_te = te; new_last_sz = b
                    new_s = s + b
                    dpt = dp[new_s]
                    key = (new_last, new_te, new_last_sz)
                    # prepare or get existing mlist
                    exist = dpt.get(key)
                    if exist is None:
                        # we'll collect into dict then to list
                        acc = {}
                    else:
                        acc = dict(exist)
                    # convolve
                    for m, mv in mlist:
                        if mv:
                            for k, cv in conv:
                                nm = m + k
                                if nm <= maxM:
                                    acc[nm] = (acc.get(nm, 0) + mv * cv) % P
                    # store back as list
                    dpt[key] = [(mm, acc[mm]) for mm in sorted(acc) if acc[mm]]
            else:
                # last==1, next is E; te+b <= half
                maxb = half - te
                a = last_sz  # prev O-size
                for b in range(1, maxb+1):
                    conv = conv_sparse.get((a, b))
                    if not conv: continue
                    new_last = 0; new_te = te + b; new_last_sz = b
                    new_s = s + b
                    dpt = dp[new_s]
                    key = (new_last, new_te, new_last_sz)
                    exist = dpt.get(key)
                    if exist is None:
                        acc = {}
                    else:
                        acc = dict(exist)
                    for m, mv in mlist:
                        if mv:
                            for k, cv in conv:
                                nm = m + k
                                if nm <= maxM:
                                    acc[nm] = (acc.get(nm, 0) + mv * cv) % P
                    dpt[key] = [(mm, acc[mm]) for mm in sorted(acc) if acc[mm]]
    # collect results from dp[N] for te=to=half
    res = [0] * (maxM+1)
    for (last, te, last_sz), mlist in dp[N].items():
        if te != half: continue
        # check to
        to = N - te
        if to != half: continue
        # add all m counts
        for m, mv in mlist:
            res[m] = (res[m] + mv) % P
    # output from M=N-1 to maxM
    out = []
    for M in range(N-1, maxM+1):
        out.append(str(res[M]))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()
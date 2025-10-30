import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline
    N,K = map(int, input().split())
    # Bucket functions by A
    buckets = {}
    for _ in range(N):
        A,B = map(int, input().split())
        buckets.setdefault(A, []).append(B)
    # Prepare buckets sorted by A descending
    # Each bucket: A, Bs sorted descending, prefix ws and powA up to K
    bucket_list = []
    for A, Bs in buckets.items():
        if not Bs:
            continue
        Bs.sort(reverse=True)
        maxc = min(len(Bs), K)
        # prefix weight sums ws[c] = sum_{k=0..c-1} Bs[k] * A^(c-1-k)
        ws = [0] * (maxc+1)
        # precompute powers of A
        powA = [1] * (maxc+1)
        for i in range(1, maxc+1):
            powA[i] = powA[i-1] * A
        # compute ws
        # for c from 1..maxc: ws[c] = sum_{k=0..c-1} Bs[k] * A^(c-1-k)
        # we can compute ws incrementally
        # But simple double loop since K<=10
        for c in range(1, maxc+1):
            s = 0
            # sum k=0..c-1
            # A power A^(c-1-k)
            exp = 1  # will use A^(c-1), then divide by A each time?
            # better precompute A_pows = powA
            for k in range(c):
                # A^(c-1-k) = powA[c-1-k]
                s += Bs[k] * powA[c-1-k]
            ws[c] = s
        bucket_list.append((A, maxc, powA, ws))
    # sort buckets by A descending
    bucket_list.sort(key=lambda x: x[0], reverse=True)
    # Global maximum A
    max_A = 0
    for A,_,_,_ in bucket_list:
        if A > max_A: max_A = A
    # Precompute max_A powers
    max_powA = [1] * (K+1)
    for i in range(1, K+1):
        max_powA[i] = max_powA[i-1] * max_A
    # Beam search DP over buckets
    # states[t] = list of (P,S) with t picks so far
    states = [[] for _ in range(K+1)]
    states[0] = [(1,0)]
    # beam width
    BEAM = 300
    for A, maxc, powA, ws in bucket_list:
        # new states
        new_states = [[] for _ in range(K+1)]
        for t in range(K+1):
            lst = states[t]
            if not lst:
                continue
            # for each state, try c picks from this bucket
            for (P,S) in lst:
                # c=0
                new_states[t].append((P,S))
                # c from 1..maxc and <=K-t
                lim = maxc
                if t + lim > K:
                    lim = K - t
                # if lim <= 0: skip
                for c in range(1, lim+1):
                    # new P, new S
                    # add = P * ws[c], P1 = P * powA[c]
                    add = P * ws[c]
                    P1 = P * powA[c]
                    S1 = S + add
                    new_states[t+c].append((P1, S1))
        # prune beam for each t
        # use upper bound est = S + P * max_A^(K-t)
        for t in range(K+1):
            lst = new_states[t]
            if len(lst) <= BEAM:
                states[t] = lst
            else:
                # compute est and select top BEAM
                # to speed, transform to (est, P, S)
                # Then sort and trim
                est_list = []
                mpow = max_powA[K-t]
                # compute est once
                for (P,S) in lst:
                    est = S + P * mpow
                    est_list.append((est, P, S))
                # partial sort: use sort and slice
                est_list.sort(key=lambda x: x[0], reverse=True)
                trimmed = est_list[:BEAM]
                # restore (P,S)
                states[t] = [(P,S) for (_,P,S) in trimmed]
    # final answer from states[K]
    res = 0
    for (P,S) in states[K]:
        total = S + P  # final residual multiply by 1
        if total > res:
            res = total
    # Edge case: if K==0, result is 1
    if K == 0:
        res = 1
    print(res)

if __name__ == "__main__":
    main()
import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        N = int(next(it)); X = int(next(it)); K = int(next(it))
        # build list of ancestors: A[0]=X, A[1]=X//2, ..., until 1
        anc = []
        u = X
        while u > 0:
            anc.append(u)
            u //= 2
        L = len(anc)
        # precompute maxd[i] = floor(log2(N//anc[i]))
        maxd = [0]*L
        for i in range(L):
            # anc[i] <= N always, so N//anc[i] >= 1
            maxd[i] = (N // anc[i]).bit_length() - 1
        total = 0
        # iterate over t = number of steps up from X to LCA
        for t, u in enumerate(anc):
            if t > K:
                break
            D = K - t
            if D < 0:
                break
            # if want to go down D from u, but D beyond possible depth, skip
            if D > maxd[t]:
                continue
            # compute number of nodes at depth D in subtree of u
            # Lnode = u * 2^D, Rnode = Lnode + (2^D - 1)
            # tot = max(0, min(Rnode, N) - Lnode + 1)
            # since D <= maxd[t], Lnode <= N
            Lnode = u << D
            span = 1 << D
            Rnode = Lnode + span - 1
            if Rnode <= N:
                tot = span
            else:
                # Rnode > N >= Lnode
                tot = N - Lnode + 1
            # subtract those in the branch we came from, if t >= 1
            if t >= 1:
                c = anc[t-1]
                Dm = D - 1
                if Dm >= 0 and Dm <= maxd[t-1]:
                    # count in subtree of c at depth Dm
                    Lc = c << Dm
                    span_c = 1 << Dm
                    Rc = Lc + span_c - 1
                    if Rc <= N:
                        sub = span_c
                    else:
                        sub = N - Lc + 1
                else:
                    sub = 0
                tot -= sub
            total += tot
        out.append(str(total))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
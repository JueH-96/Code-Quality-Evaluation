def main():
    import sys
    input = sys.stdin.readline

    N, X, Y = map(int, input().split())
    P = [0] * (N)
    T = [0] * (N)
    # we only need P[1..N-1], T[1..N-1]
    for i in range(1, N):
        pi, ti = map(int, input().split())
        P[i] = pi
        T[i] = ti

    # LCM of 1..8 = 840
    L = 840
    # D[r] = total extra time after all buses (from stop1 to stopN)
    # when you arrive at stop1 at time t with t mod L = r,
    # then after taking buses 1..N-1 you gain D[r] over t.
    D = [0] * L

    # Build D by folding each bus stop i=1..N-1
    # h_{k-1}(t) = t + D[t mod L]
    # then f_k applied: wait = (- (h_{k-1}(t) mod P_k)) mod P_k
    #                = (P_k - ((r + D[r]) % P_k)) % P_k
    # so D_new[r] = D[r] + wait + T_k
    # We update in place for k=1..N-1
    for i in range(1, N):
        pi = P[i]
        ti = T[i]
        # speed up locals
        Di = D
        m = pi
        addt = ti
        # for each residue r
        # compute s = (r + Di[r]) % m
        # w = (m - s) % m
        # Di[r] += w + addt
        for r in range(L):
            # compute remainder mod m
            s = (r + Di[r]) % m
            if s:
                w = m - s
                Di[r] += w + addt
            else:
                Di[r] += addt
        # continue with updated D

    Q = int(input())
    out = []
    for _ in range(Q):
        q = int(input())
        # walk to stop1: +X
        t0 = q + X
        # apply bus chain: delta = D[t0 % L]
        delta = D[t0 % L]
        # walk from last stop: +Y
        out.append(str(t0 + delta + Y))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()
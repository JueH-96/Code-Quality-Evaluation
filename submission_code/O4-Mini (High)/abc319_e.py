import sys
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    # Read inputs
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    P = [0] * (N-1)
    T = [0] * (N-1)
    for i in range(N-1):
        P[i] = int(next(it))
        T[i] = int(next(it))
    Q = int(next(it))
    qs = [0]*Q
    for i in range(Q):
        qs[i] = int(next(it))
    # Compute full modulus = lcm of all P_i
    from math import gcd
    M_full = 1
    for p in P:
        if p > 1:
            M_full = M_full // gcd(M_full, p) * p
    # Precompute remainders j % m for each divisor m of M_full
    # First get divisors of M_full
    divs = []
    mf = M_full
    i = 1
    # simple divisor enumeration up to sqrt
    import math
    rlim = int(math.isqrt(mf))
    for a in range(1, rlim+1):
        if mf % a == 0:
            divs.append(a)
            b = mf // a
            if b != a:
                divs.append(b)
    # ensure sorted for consistency (not strictly needed)
    # but we need fast lookup, so build dict
    rem_table = {}
    for m in divs:
        # only for m >= 1 and <= M_full
        # build list of j%m for j in 0..M_full-1
        arr = [0]*M_full
        if m != 0:
            for j in range(m):
                arr[j] = j
            # fill the rest
            # arr[j] = j % m  for j in m..M_full-1
            # pattern repeats every m
            # we can copy blocks
            for base in range(m, M_full, m):
                end = base + m
                if end > M_full:
                    end = M_full
                length = end - base
                # copy arr[0:length] to arr[base:base+length]
                arr[base:end] = arr[0:length]
        # else m==0 won't happen since divisors positive
        rem_table[m] = arr
    # Now compose bus-stop functions
    M = 1
    d = [0]
    # locals for speed
    table = rem_table
    for pi, ti in zip(P, T):
        if pi == 1:
            # always no wait, just add ti
            if ti:
                # in-place add
                dj = d
                for j in range(M):
                    dj[j] += ti
            # modulus stays same
            continue
        # pi > 1
        g = gcd(M, pi)
        M2 = (M // g) * pi
        p = pi
        t = ti
        if M2 == M:
            # update in place on d[0..M-1]
            dj = d
            rem_p = table[p]
            mloc = M
            # localize p,t
            p_loc = p
            t_loc = t
            for j in range(mloc):
                old = dj[j]
                # arrival mod p
                rem = (rem_p[j] + old) % p_loc
                # wait time
                if rem:
                    w = p_loc - rem
                else:
                    w = 0
                dj[j] = old + w + t_loc
        else:
            # build new d2 of length M2
            rem_p = table[p]
            rem_old = table[M]
            mloc = M
            d2 = [0] * M2
            p_loc = p
            t_loc = t
            # fill d2
            for j in range(M2):
                old = d[ rem_old[j] ]
                rem = (rem_p[j] + old) % p_loc
                if rem:
                    w = p_loc - rem
                else:
                    w = 0
                d2[j] = old + w + t_loc
            d = d2
            M = M2
    # Now answers: for each query qi, time = qi + X + d[(qi+X)%M] + Y
    C = X + Y
    out = []
    dloc = d
    Mloc = M
    for qi in qs:
        r = (qi + X) % Mloc
        out.append(str(qi + C + dloc[r]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
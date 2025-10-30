def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast iterator over the input tokens
    it = 0
    def read_int():
        nonlocal it
        val = int(input_data[it])
        it += 1
        return val

    N = read_int()
    X = read_int()
    Y = read_int()

    P = []
    T = []
    for _ in range(N - 1):
        p_i = read_int()
        t_i = read_int()
        P.append(p_i)
        T.append(t_i)

    Q = read_int()
    queries = [read_int() for _ in range(Q)]

    # If there are no bus rides (the problem constraints have N >= 2, so there's at least 1 bus ride)
    # but let's just be cautious:
    if N == 1:
        # Then we only walk: house -> bus stop 1 (which doesn't exist) -> Aoki's
        # So time = q + X + Y for each query
        import sys
        out = []
        for q in queries:
            out.append(str(q + X + Y))
        print("
".join(out))
        return

    # Compute the LCM of all P_i (each p_i <= 8)
    # LCM of numbers up to 8 is at most 840.
    # We'll build a function that from time t -> time after all bus rides R(t),
    # then final F(t) = R(t + X) + Y.  But R(t + LCM) = R(t) + LCM*(N-1).
    # So we only need to precompute R(u) for u in [0..LCM-1], then use the periodicity.
    from math import gcd
    def lcm(a,b):
        return a*b // gcd(a,b)

    LCM = 1
    for p in P:
        LCM = lcm(LCM, p)

    # Precompute R(u) for u in [0..LCM-1].
    # We'll store it in an array cur[], where cur[u] will become R(u)
    # by repeatedly applying nextMultiple(t, P[i]) + T[i].
    cur = list(range(LCM))

    # Apply each bus ride in order
    # nextMultiple(x, p) = x if x%p == 0 else x + (p - x%p).
    # A fast way in Python is x += -x % p.
    for i in range(N - 1):
        p = P[i]
        t_add = T[i]
        for idx in range(LCM):
            x = cur[idx]
            # wait enough so that x is multiple of p
            off = (-x) % p
            x += off
            # ride time
            x += t_add
            cur[idx] = x

    # Now cur[u] = R(u). For the final arrival, we add Y (walk from stop N to Aoki's).
    # So the full function if leaving house at time q is F(q) = R(q+X) + Y.
    # Then we use the periodicity: R(u + k*LCM) = R(u) + k*LCM*(N-1).
    # So F(q) = R((q+X)%LCM) + Y + ((q+X)//LCM)*LCM*(N-1).
    # We'll fold the +Y into cur[u] to avoid adding it repeatedly.
    for idx in range(LCM):
        cur[idx] += Y

    # Now answer queries
    out = []
    base_shift = LCM * (N - 1)
    for q in queries:
        # s = (q+X) mod LCM
        # k = (q+X) // LCM
        # answer = cur[s] + k * base_shift
        q_plus_x = q + X
        s = q_plus_x % LCM
        k = q_plus_x // LCM
        ans = cur[s] + k * base_shift
        out.append(str(ans))

    print("
".join(out))

# Call solve() after defining it
solve()
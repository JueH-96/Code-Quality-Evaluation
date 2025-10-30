import sys
import threading
def main():
    import sys
    data = sys.stdin
    # Read inputs
    N_X_Y = data.readline().split()
    while len(N_X_Y) < 3:
        N_X_Y += data.readline().split()
    N = int(N_X_Y[0])
    X = int(N_X_Y[1])
    Y = int(N_X_Y[2])
    P = [0] * (N-1)
    T = [0] * (N-1)
    for i in range(N-1):
        pi_ti = data.readline().split()
        while len(pi_ti) < 2:
            pi_ti += data.readline().split()
        P[i] = int(pi_ti[0])
        T[i] = int(pi_ti[1])
    Q = int(data.readline())
    qs = [0]*Q
    for i in range(Q):
        qs[i] = int(data.readline())
    # Compute period M = lcm of all P_i
    from math import gcd
    M = 1
    for pi in P:
        M = M // gcd(M, pi) * pi
    # Precompute D[m] = total bus-travel time from stop1 to stopN,
    # starting exactly at time m (0 <= m < M) at stop1.
    D = [0] * M
    P_list = P  # local
    T_list = T
    Nm1 = N - 1
    # For speed, bring locals
    for m in range(M):
        t = m
        # simulate through all bus segments
        for i in range(Nm1):
            pi = P_list[i]
            # wait until next multiple of pi
            r = t % pi
            if r:
                t += pi - r
            # ride time
            t += T_list[i]
        # store net added time
        D[m] = t - m
    # Process queries
    out = []
    add_walk = X  # time from house to stop1
    for q in qs:
        start = q + add_walk
        mod = start % M
        ans = q + add_walk + D[mod] + Y
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
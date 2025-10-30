def main():
    import sys, numpy as np
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    M = N - 1  # there are N-1 bus rides
    # Read the bus parameters for bus stops 1,...,N-1.
    P_list = [0] * M
    T_list = [0] * M
    for i in range(M):
        P_list[i] = int(next(it))
        T_list[i] = int(next(it))
    Q = int(next(it))
    queries = [int(next(it)) for _ in range(Q)]
    
    # Key observation:
    # Each bus ride from bus stop i to (i+1) works as follows.
    #   If Takahashi arrives at bus stop i at time t, then he must wait
    #   until a departure time which is a multiple of P_i. In other words,
    #   he catches a bus at time t' = ceil(t / P_i) * P_i.
    #   Then he spends additional T_i units to reach bus stop (i+1).
    # Notice that
    #   ceil(t / P_i) * P_i = t + ((-t) mod P_i)
    # so the bus ride “function” is
    #   f_i(t) = t + ( (-t)%P_i + T_i ).
    #
    # Important: Since 1 <= P_i <= 8, we observe that each P_i divides 840,
    # which is the least common multiple of 1,2,...,8. In other words, if we
    # write t = L*k + r with L = 840 and 0 <= r < L, then because P_i divides L
    # we have (t mod P_i) = (r mod P_i) and so:
    #   f_i(t) = t + ( (-r)%P_i + T_i ).
    #
    # Thus when applying a bus ride there is an “extra delay” that depends only
    # on r, the residue of t modulo L. And composing these bus functions yields a function
    # F(s) = s + D(s mod L) (i.e. an additive “delay” depending on the starting residue).
    #
    # Our plan is to compute (by dynamic programming in “backward”
    # order) an array dp[0..L-1] so that, if Takahashi arrives at bus stop 1
    # with time t (and r = t mod L), then the overall delay (over all N-1 bus rides)
    # is dp[r]. Then because he walks from his house to bus stop 1 in time X and from bus stop N
    # to Aoki's house in time Y, the answer for query q (departure time from his house)
    # will be:
    # 
    #   answer = (q + X) + dp[(q + X) mod L] + Y
    #
    # We use numpy arrays with L = 840 states. There are M bus rides and for each ride
    # we update our dp function. (Working from the final bus ride backwards.)
    
    L = 840
    dp = np.zeros(L, dtype=np.int64)  # dp[r] = extra delay needed when you are at a stop with time ≡ r mod L.
    r_arr = np.arange(L, dtype=np.int64)  # represents residues 0,1,...,L-1

    # Process the bus rides in reverse order.
    # For a given bus ride with parameters P and T, if you arrive with time ≡ r (mod L),
    # then your waiting time is: wait = (-r) mod P  and you ride and add T,
    # so the extra delay from that ride is (wait + T) plus the delay computed from the next ride.
    # And the new residue becomes: (r + wait + T) mod L.
    for P_val, T_val in zip(reversed(P_list), reversed(T_list)):
        # Compute waiting time vector: for each residue r, wait = (-r) mod P_val.
        w = (-r_arr) % P_val
        add = w + T_val  # additional delay in this bus ride
        new_r = (r_arr + add) % L  # new residue after the bus ride
        dp = add + dp[new_r]  # update dp function; note numpy indexing applies elementwise

    # Now answer each query.
    # Takahashi’s schedule:
    #   - He waits until his query departure time q,
    #   - Walks to bus stop 1 in time X, arriving at time (q + X).
    #   - Then uses the buses adding delay dp[(q+X)%L],
    #   - Then walks from stop N to Aoki’s house in Y extra time.
    out_lines = []
    for q in queries:
        start = q + X
        r = start % L
        ans = start + int(dp[r]) + Y
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()
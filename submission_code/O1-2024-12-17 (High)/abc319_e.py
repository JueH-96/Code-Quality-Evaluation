def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast iterator for the input data
    it = 0
    def input_int():
        nonlocal it
        val = int(input_data[it])
        it += 1
        return val

    # Read N, X, Y
    N = input_int()
    X = input_int()
    Y = input_int()

    # Read P_i, T_i
    P = [0]*(N-1)
    T = [0]*(N-1)
    for i in range(N-1):
        P[i] = input_int()
        T[i] = input_int()

    # Compute L = lcm of all P_i
    # Since P_i <= 8, L will be at most 840
    def gcd(a,b):
        while b:
            a,b = b,a%b
        return a
    def lcm(a,b):
        return a//gcd(a,b)*b

    L = 1
    for i in range(N-1):
        L = lcm(L, P[i])

    # We build from right (stop N) backward to left (stop 1).
    # Let H_i(t) = arrival time at stop N if we arrive at stop i at time t.
    # Then H_N(t) = t (no more buses), so define G_N(r) = H_N(r) - r = 0 for all r.
    # Recurrence:
    #   H_i(t) = H_{i+1}( nextDepart(t,P_i) + T_i )
    # => G_i(r) = [ nextDepart(r,P_i) + T_i ] + G_{i+1}(... mod L) - r
    # We'll store G in an array of length L for each step, but only keep one array
    # and update it in-place from i = N-2 down to 0.

    G = [0]*L  # This corresponds to G_{N}(r) initially (all zero)

    # Build G_i from right to left
    for i in range(N-2, -1, -1):
        p = P[i]
        t_i = T[i]
        # Reduce T[i] mod L to speed up
        t_mod = t_i % L

        newG = [0]*L
        # Compute newG[r] = offset + G[(r + offset) mod L]
        # where offset = t_mod + (0 if r%p==0 else p - (r%p))
        for r in range(L):
            r_p = r % p  # remainder mod p
            w = 0 if r_p == 0 else (p - r_p)  # waiting
            offset = t_mod + w

            nr = r + offset
            # since offset <= L + (p-1), we can subtract L at most twice
            if nr >= L:
                nr -= L
            if nr >= L:
                nr -= L

            newG[r] = offset + G[nr]

        G = newG

    # Now G corresponds to G_1(r).
    # If Takahashi arrives at bus stop 1 at time t, the arrival time at stop N is t + G[t mod L].
    # Finally he walks Y to Aoki's house, so final = t + G[t mod L] + Y.
    # But from home to stop1 is X, so if he leaves home at q, t = q + X.

    Q = input_int()
    out = []
    for _ in range(Q):
        q = input_int()
        t = q + X
        r = t % L
        ans = t + G[r] + Y
        out.append(str(ans))

    # Print all answers
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()
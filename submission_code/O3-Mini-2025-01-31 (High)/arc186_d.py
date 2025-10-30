# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    # read input: first integer N then N numbers A1...A_N
    try:
        N = int(data[0])
    except:
        return
    A = list(map(int, data[1:]))

    mod = 998244353

    # Special case: for N==1 the only valid Polish sequence is (0).
    # (0) is ≤ A if 0 <= A[0] so answer is 1.
    if N == 1:
        sys.stdout.write("1")
        return

    # Precompute factorials and inverse factorials up to 2*N.
    max_n = 2 * N
    fact = [1] * (max_n + 1)
    invfact = [1] * (max_n + 1)
    for i in range(2, max_n+1):
        fact[i] = fact[i-1] * i % mod
    invfact[max_n] = pow(fact[max_n], mod-2, mod)
    for i in range(max_n, 0, -1):
        invfact[i-1] = invfact[i] * i % mod

    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n] * invfact[r] % mod * invfact[n-r] % mod

    # sumF(L, a, b): For L>=1, computes
    #   S = sum_{t=a}^{b} (t/L) * binom(2L-t-1, L-1)
    # with a closed-form: Let X = 2L-1, U_max = X-a, U_min = X-b.
    # Then S = 2*( nCr(U_max+1, L) - nCr(U_min, L) )
    #      - ( nCr(U_max+2, L+1) - nCr(U_min+1, L+1) ).
    def sumF(L, a, b):
        if a > b:
            return 0
        X = 2 * L - 1
        U_max = X - a
        U_min = X - b
        A_term = (nCr(U_max + 1, L) - nCr(U_min, L)) % mod
        B_term = (nCr(U_max + 2, L+1) - nCr(U_min + 1, L+1)) % mod
        return (2 * A_term - B_term) % mod

    # Now we do a DP along the positions.
    # Initially, state r = 1.
    # For positions p = 0,...,N-2, let rem = N - p and allowed x satisfy:
    #    x_min = 1 if r==1 else 0, and x_max = rem - r.
    # We add contributions (via sumF) from choices x from x_min to min(x_max, A[p]-1).
    # In the tight branch we require A[p] is in [x_min, x_max] and update r = r + A[p] - 1.
    # For p = N-1, we need r == 1 and A[N-1] == 0.
    ans = 0
    r = 1
    valid = True
    for p in range(N):
        rem = N - p   # number of digits remaining (including current)
        if p < N - 1:
            L_rem = rem - 1  # positions to complete after current
            if r == 1:
                x_min = 1
            else:
                x_min = 0
            x_max = rem - r
            cur = A[p]
            # Count all choices x in [x_min, min(x_max, cur-1)] if any.
            if cur > x_min:
                cand_max = cur - 1
                if cand_max > x_max:
                    cand_max = x_max
                if cand_max >= x_min:
                    t_low = r + x_min - 1
                    t_high = r + cand_max - 1
                    if t_high > L_rem:
                        t_high = L_rem
                    if t_low <= t_high:
                        ans = (ans + sumF(L_rem, t_low, t_high)) % mod
            if cur < x_min or cur > x_max:
                valid = False
                break
            r = r + cur - 1
        else:
            cur = A[p]
            if r != 1 or cur != 0:
                valid = False
                break
            else:
                ans = (ans + 1) % mod
    sys.stdout.write(str(ans % mod))
    
if __name__ == '__main__':
    main()
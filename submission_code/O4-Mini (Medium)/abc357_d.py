def main():
    import sys
    input_data = sys.stdin.readline().strip()
    MOD = 998244353

    # Read N as string to get its length L, and also parse as integer
    N_str = input_data
    N = int(N_str)
    L = len(N_str)

    # N modulo MOD
    N_mod = N % MOD

    # r = 10^L mod MOD
    r = pow(10, L, MOD)

    # Compute S = 1 + r + r^2 + ... + r^(N-1)  (mod MOD)
    # If r != 1, geometric series formula
    if r != 1:
        # numerator = r^N - 1  (mod MOD)
        num = pow(r, N, MOD) - 1
        if num < 0:
            num += MOD
        # inv_den = inverse of (r - 1) mod MOD
        inv_den = pow(r - 1, MOD - 2, MOD)
        S = num * inv_den % MOD
    else:
        # r == 1 => each term is 1, so sum is N mod MOD
        S = N_mod

    # V_N mod MOD = N * S mod MOD
    ans = N_mod * S % MOD

    print(ans)


if __name__ == "__main__":
    main()
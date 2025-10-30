def main():
    s = input().strip()
    N = int(s)
    MOD = 998244353

    # A is the integer value of N mod MOD
    A = N % MOD
    # L is the number of digits of N
    L = len(s)
    # r = 10^L mod MOD
    r = pow(10, L, MOD)

    # We need S = 1 + r + r^2 + ... + r^(N-1) mod MOD
    # If r != 1 mod MOD, S = (r^N - 1) / (r - 1) mod MOD
    if r == 1:
        S = N % MOD
    else:
        numerator = (pow(r, N, MOD) - 1) % MOD
        denom = (r - 1) % MOD
        inv_denom = pow(denom, MOD - 2, MOD)
        S = numerator * inv_denom % MOD

    # The final answer is A * S mod MOD
    print(A * S % MOD)

if __name__ == "__main__":
    main()
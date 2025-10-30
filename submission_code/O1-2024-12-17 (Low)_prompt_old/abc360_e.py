def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    MOD = 998244353

    # If N = 1, there's only one ball (the black ball),
    # it always stays in position 1, so the answer is 1.
    if N == 1:
        print(1)
        return

    # Modular inverse function (using Fermat's little theorem, since MOD is prime)
    def modinv(a, m=MOD):
        return pow(a, m - 2, m)

    invN = modinv(N)          # Inverse of N mod MOD
    inv2 = modinv(2)          # Inverse of 2 mod MOD

    # Recurrence for E[X_{t+1}] given E[X_t] is:
    #   E[X_{t+1}] = (N-2)/N * E[X_t] + (N+1)/N
    # Solve its closed form:
    #   E[K] = (N+1)/2 + ((N-2)/N)^K * (1 - N)/2
    # We'll compute this in modular arithmetic.

    # L = (N+1)/2 (mod MOD)
    L = (N + 1) % MOD
    L = (L * inv2) % MOD

    # diff = (1 - N)/2 (mod MOD)
    # First compute (1 - N) mod, then multiply by inv2.
    diff = (1 - N) % MOD
    diff = (diff * inv2) % MOD

    # alpha = (N-2)/N (mod MOD)
    alpha_num = (N - 2) % MOD  # numerator
    alpha = (alpha_num * invN) % MOD

    # alpha^K
    alpha_pow_k = pow(alpha, K, MOD)

    # Final E[K] = L + alpha^K * diff (all mod)
    ans = (L + alpha_pow_k * diff) % MOD

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()
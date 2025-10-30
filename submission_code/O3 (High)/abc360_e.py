import sys

MOD = 998244353
INV2 = (MOD + 1) // 2          # 2^{-1}  (mod MOD)

def main() -> None:
    N_str, K_str = sys.stdin.read().split()
    N = int(N_str)
    K = int(K_str)

    n = N % MOD                 # N itself is < MOD but keep it explicit
    inv_n = pow(n, MOD - 2, MOD)        # modular inverse of N
    a = (n - 2) % MOD * inv_n % MOD     # a = (N-2)/N = 1 - 2/N (mod MOD)
    a_pow = pow(a, K, MOD)              # a^K (mod MOD)

    # E_K  = (N+1 - (N-1) * a^K) / 2   (mod MOD)
    numerator = (n + 1 - (n - 1) * a_pow) % MOD
    answer = numerator * INV2 % MOD
    print(answer)


if __name__ == "__main__":
    main()
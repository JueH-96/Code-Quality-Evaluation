import sys

MOD = 998244353        # given prime
PHI = MOD - 1          # Euler's totient (since MOD is prime)


def main() -> None:
    n_str = sys.stdin.readline().strip()
    if not n_str:                       # in case of empty line
        return
    n = int(n_str)
    d = len(n_str)                      # number of decimal digits of N

    r = pow(10, d, MOD)                 # r = 10^d (mod MOD)
    denom = (r - 1) % MOD               # denominator of the geometric sum

    # geometric series  S = 1 + r + r^2 + ... + r^{N-1}  (mod MOD)
    if denom != 0:                      # usual case
        exp = (d * (n % PHI)) % PHI     # d * N  reduced modulo PHI
        numerator = (pow(10, exp, MOD) - 1) % MOD        # 10^{dN} - 1 (mod)
        inv_denom = pow(denom, MOD - 2, MOD)             # modular inverse
        S = (numerator * inv_denom) % MOD
    else:                               # when 10^d == 1 (mod MOD)
        # every term of the series is 1, so S == N (mod MOD)
        S = n % MOD

    result = (n % MOD) * S % MOD        # V_N = N * S   (mod MOD)
    print(result)


if __name__ == "__main__":
    main()
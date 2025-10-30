import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    MOD = 998244353

    # Number of digits in N
    d = len(data)

    # Compute 10^d mod MOD
    pow10d = pow(10, d, MOD)

    # Compute the geometric series S = 1 + 10^d + 10^{2d} + ... + 10^{(N-1)d} mod MOD
    # If pow10d == 1 mod MOD, then S = N mod MOD
    if pow10d == 1:
        S = N % MOD
    else:
        numerator = (pow(pow10d, N, MOD) - 1) % MOD
        denom = (pow10d - 1) % MOD
        inv_denom = pow(denom, MOD - 2, MOD)
        S = numerator * inv_denom % MOD

    # V_N = N * S
    result = (N % MOD) * S % MOD
    print(result)

if __name__ == "__main__":
    main()
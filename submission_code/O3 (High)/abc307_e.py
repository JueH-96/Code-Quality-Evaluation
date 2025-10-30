import sys

MOD = 998244353

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, M = map(int, data)

    # x is (M-1) modulo MOD
    x = (M - 1) % MOD

    # (M-1)^N  (mod MOD)
    pow_part = pow(x, N, MOD)

    # (+/-)(M-1)  depending on parity of N
    if N % 2 == 0:
        sign_part = x          # (+1)*(M-1)
    else:
        sign_part = (MOD - x)  # (-1)*(M-1)  modulo MOD

    ans = (pow_part + sign_part) % MOD
    print(ans)

if __name__ == "__main__":
    main()
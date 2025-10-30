MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    inv3 = (332748118)  # since 3 * 332748118 ≡ 1 mod 998244353

    a = N % MOD
    b = M % MOD
    c = (M + 1) % MOD
    d = (2 * N + 1) % MOD

    res = a * b % MOD
    res = res * c % MOD
    res = res * d % MOD
    res = res * inv3 % MOD

    print(res)

if __name__ == '__main__':
    main()
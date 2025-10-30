MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    inv6 = pow(6, MOD-2, MOD)
    res = N * N % MOD
    res = res * M % MOD
    res = res * (M + 1) % MOD
    res = res * (2 * M + 1) % MOD
    res = res * inv6 % MOD
    print(res)

if __name__ == "__main__":
    main()
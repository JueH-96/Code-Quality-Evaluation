def main():
    import sys
    MOD = 998244353
    N = int(sys.stdin.read())
    d = len(str(N))
    base = pow(10, d, MOD)
    S = (pow(base, N, MOD) - 1) * pow(base - 1, MOD - 2, MOD) % MOD
    VN = (N * S) % MOD
    print(VN)

if __name__ == "__main__":
    main()
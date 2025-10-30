MOD = 998244353

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    term1 = pow(m - 1, n, MOD)
    sign = 1 if n % 2 == 0 else -1
    term2 = (sign * (m - 1)) % MOD
    ans = (term1 + term2) % MOD
    print(ans)

if __name__ == "__main__":
    main()
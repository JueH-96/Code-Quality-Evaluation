mod = 998244353

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    base = pow(m - 1, n, mod)
    if n % 2 == 0:
        ans = (base + (m - 1)) % mod
    else:
        ans = (base - (m - 1)) % mod
    print(ans)

if __name__ == "__main__":
    main()
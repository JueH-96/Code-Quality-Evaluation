MOD = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    base = m - 1
    term1 = pow(base, n, MOD)
    if n % 2 == 0:
        ans = (term1 + base) % MOD
    else:
        ans = (term1 + MOD - base) % MOD
    print(ans)

if __name__ == "__main__":
    main()
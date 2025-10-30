MOD = 998244353

def solve(n, m):
    ans = 0
    while n > 0:
        n &= n - 1
        ans += m.bit_count()
        m &= m - 1
        ans %= MOD
    return ans

def main():
    n, m = map(int, input().split())
    ans = 0
    for i in range(61):
        if m >> i & 1:
            ans += (1 << i) * pow(2, n - (1 << i) + 1, MOD) % MOD * pow(2, i, MOD) % MOD
            ans %= MOD
    print(ans)

if __name__ == "__main__":
    main()
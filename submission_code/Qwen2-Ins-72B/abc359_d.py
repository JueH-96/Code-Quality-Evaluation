MOD = 998244353

def pow_mod(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return x * pow_mod(x, n - 1) % MOD
    else:
        t = pow_mod(x, n // 2)
        return t * t % MOD

def solve(s, k):
    n = len(s)
    dp = [0] * (1 << k)
    dp[0] = 1
    for i in range(n):
        ndp = [0] * (1 << k)
        for j in range(1 << k):
            if bin(j).count('1') * 2 == k:
                continue
            if s[i] != 'A':
                ndp[(j << 1) % (1 << k) | 1] += dp[j]
                ndp[(j << 1) % (1 << k) | 1] %= MOD
            if s[i] != 'B':
                ndp[(j << 1) % (1 << k)] += dp[j]
                ndp[(j << 1) % (1 << k)] %= MOD
        dp = ndp
    return sum(dp) % MOD

def main():
    n, k = map(int, input().split())
    s = input()
    ans = pow_mod(2, s.count('?'))
    for i in range(k, len(s) + 1):
        t = s[i - k:i]
        if '?' not in t:
            if t == t[::-1]:
                ans = (ans - solve(t, k) + MOD) % MOD
    print(ans)

if __name__ == '__main__':
    main()
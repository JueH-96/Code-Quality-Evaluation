def solve():
    n, m = map(int, input().split())
    mod = 998244353

    primes = [2, 3, 5, 7, 11, 13]
    divisors = [0] * (m + 1)
    for i in range(1, m + 1):
        divisors[i] = 1
        x = i
        for p in primes:
            cnt = 0
            while x % p == 0:
                cnt += 1
                x //= p
            divisors[i] *= (cnt + 1)

    dp = {}
    def get_dp(k):
        if k == 0:
            return 1
        if k in dp:
            return dp[k]
        
        res = 0
        if k % 2 == 0:
            half = get_dp(k // 2)
            res = (half * half) % mod
        else:
            res = (get_dp(k - 1) * sum(divisors[1:]) % mod) % mod
        
        dp[k] = res
        return res

    ans = 0
    for i in range(1, min(n + 1, 61)):
        ans = (ans + get_dp(i)) % mod
    
    if n > 60:
        val = get_dp(60)
        s = sum(divisors[1:]) % mod
        
        num = (pow(s, n - 60, mod) - 1 + mod) % mod
        den = (s - 1 + mod) % mod
        
        ans = (ans + val * num * pow(den, mod - 2, mod) % mod) % mod

    print(ans)

solve()
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    total_outcomes = 1
    for x in a:
        total_outcomes = (total_outcomes * x) % mod

    dp = [0] * 11
    dp[0] = 1

    for die_max in a:
        new_dp = dp[:]
        for i in range(1, 11):
            for j in range(1, min(die_max + 1, 11)):
                if i - j >= 0:
                    new_dp[i] = (new_dp[i] + dp[i-j]) % mod
        dp = new_dp
    
    
    successful_outcomes = (total_outcomes - dp[0]) % mod
    
    
    def power(base, exp, m):
        res = 1
        base %= m
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % m
            base = (base * base) % m
            exp //= 2
        return res

    def mod_inverse(a, m):
        return power(a, m - 2, m)

    
    result = (successful_outcomes * mod_inverse(total_outcomes, mod)) % mod
    print(result)

solve()
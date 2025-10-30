def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    expected_salary = 0
    
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        prob_reach_i = 0
        for j in range(i):
            prob_reach_i = (prob_reach_i + dp[j]) % mod
        
        prob_reach_i = (prob_reach_i + 1) % mod
        
        prob_stay = pow(n, mod - 2, mod)
        
        dp[i] = (prob_reach_i * prob_stay) % mod
        
        expected_salary = (expected_salary + dp[i] * a[i-1]) % mod
        
    print(expected_salary)

solve()
def solve():
    n, x = map(int, input().split())
    t_durations = list(map(int, input().split()))
    mod = 998244353
    
    def power(a, b, m):
        res = 1
        a %= m
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % m
            a = (a * a) % m
            b //= 2
        return res
        
    def inverse(n_val, m):
        return power(n_val, m - 2, m)
        
    inv_n = inverse(n, mod)
    
    dp = [0] * (x + 1)
    dp[0] = 1
    
    for current_time in range(1, x + 1):
        for duration in t_durations:
            if current_time >= duration:
                dp[current_time] = (dp[current_time] + dp[current_time - duration] * inv_n) % mod
                
    sum_prob = 0
    start_time = max(0, x - t_durations[0] + 1)
    for current_time in range(start_time, x + 1):
        sum_prob = (sum_prob + dp[current_time]) % mod
        
    result = (sum_prob * inv_n) % mod
    print(result)

if __name__ == '__main__':
    solve()
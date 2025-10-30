def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    mod = 998244353
    
    max_duration_sum = x
    max_songs_count = x + 2 # slightly larger than needed, but safe
    
    dp = [[0] * (max_duration_sum + 1) for _ in range(max_songs_count + 1)]
    dp[0][0] = 1
    
    for k in range(1, max_songs_count + 1):
        for s in range(max_duration_sum + 1):
            for song_duration in t:
                if s >= song_duration:
                    dp[k][s] = (dp[k][s] + dp[k-1][s - song_duration]) % mod
                    
    total_probability = 0
    inv_n = pow(n, mod - 2, mod)
    
    for k in range(1, max_songs_count + 1):
        sum_val = 0
        start_sum = max(0, x - t[0] + 1)
        for s in range(start_sum, x + 1):
            sum_val = (sum_val + dp[k-1][s]) % mod
        
        prob_k = (sum_val * pow(inv_n, k, mod)) % mod
        total_probability = (total_probability + prob_k) % mod
        
    print(total_probability)

if __name__ == '__main__':
    solve()
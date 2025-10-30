def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    mod = 998244353

    max_t = max(t)
    dp = [0] * (x + 1)
    dp[0] = 1
    inv_n = pow(n, mod - 2, mod)

    for current_time in range(1, x + 1):
        for i in range(n):
            if current_time - t[i] >= 0:
                dp[current_time] = (dp[current_time] + dp[current_time - t[i]] * inv_n) % mod

    prob = 0
    start_time = int(x + 0.5 - t[0]) + 1
    end_time = int(x)

    for start in range(start_time, end_time + 1):
        if start >= 0 and start <= x:
            prob_start_song1 = (dp[start] * inv_n) % mod
            prob = (prob + prob_start_song1) % mod

    print(prob)

solve()
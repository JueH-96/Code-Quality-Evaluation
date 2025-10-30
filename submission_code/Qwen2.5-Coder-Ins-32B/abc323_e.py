# YOUR CODE HERE
MOD = 998244353

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    T = list(map(int, input[2:]))
    
    total_time = sum(T)
    target_time = X + 0.5
    
    dp = [0] * (total_time * N + 1)
    dp[0] = 1
    
    for _ in range(N):
        new_dp = [0] * (total_time * N + 1)
        for t in range(total_time * N):
            if dp[t] == 0:
                continue
            for i in range(N):
                new_dp[t + T[i]] += dp[t]
                new_dp[t + T[i]] %= MOD
        dp = new_dp
    
    numerator = 0
    for t in range(int(target_time * 2), total_time * N + 1):
        if (t - int(target_time * 2)) % 2 == 0:
            numerator += dp[t]
            numerator %= MOD
    
    denominator = pow(N, N, MOD)
    result = (numerator * mod_inverse(denominator, MOD)) % MOD
    print(result)

solve()
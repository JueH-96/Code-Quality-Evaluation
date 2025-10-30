# YOUR CODE HERE
MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][j] will be the number of ways to get sum j using the first i dice
    dp = [[0] * 11 for _ in range(N + 1)]
    dp[0][0] = 1
    
    for i in range(1, N + 1):
        for j in range(11):
            dp[i][j] = dp[i - 1][j]
            if j >= 1:
                dp[i][j] += dp[i - 1][j - 1]
            if j >= 2:
                dp[i][j] += dp[i - 1][j - 2]
            if j >= 3:
                dp[i][j] += dp[i - 1][j - 3]
            if j >= 4:
                dp[i][j] += dp[i - 1][j - 4]
            if j >= 5:
                dp[i][j] += dp[i - 1][j - 5]
            if j >= 6:
                dp[i][j] += dp[i - 1][j - 6]
            if j >= 7:
                dp[i][j] += dp[i - 1][j - 7]
            if j >= 8:
                dp[i][j] += dp[i - 1][j - 8]
            if j >= 9:
                dp[i][j] += dp[i - 1][j - 9]
            if j >= 10:
                dp[i][j] += dp[i - 1][j - 10]
            dp[i][j] %= MOD
    
    total_ways = 1
    for a in A:
        total_ways = total_ways * a % MOD
    
    successful_ways = dp[N][10]
    
    probability = successful_ways * modinv(total_ways, MOD) % MOD
    
    print(probability)

if __name__ == "__main__":
    solve()
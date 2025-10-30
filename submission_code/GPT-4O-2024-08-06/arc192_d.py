# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][g] will store the sum of scores of sequences ending at i with gcd g
    dp = [{} for _ in range(N + 1)]
    dp[0][1] = 1  # Starting with gcd 1
    
    # Iterate over each position
    for i in range(1, N):
        a = A[i - 1]
        for g in dp[i - 1]:
            score = dp[i - 1][g]
            for d in range(1, int(a**0.5) + 1):
                if a % d == 0:
                    # d is a divisor
                    d1 = d
                    d2 = a // d
                    for d_ in [d1, d2]:
                        new_g = g * d_
                        if new_g not in dp[i]:
                            dp[i][new_g] = 0
                        dp[i][new_g] = (dp[i][new_g] + score * d_) % MOD
    
    # Calculate the sum of scores of all sequences with gcd 1 at the end
    result = 0
    for g in dp[N - 1]:
        if g == 1:
            result = (result + dp[N - 1][g]) % MOD
    
    print(result)
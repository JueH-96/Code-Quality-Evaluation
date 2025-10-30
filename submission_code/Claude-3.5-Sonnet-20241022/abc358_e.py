MOD = 998244353

def solve():
    K = int(input())
    C = list(map(int, input().split()))
    
    # dp[i][j] = number of valid strings of length j using letters up to i
    dp = [[0]*(K+1) for _ in range(27)]
    dp[0][0] = 1
    
    # For each letter
    for i in range(26):
        # For each current length
        for j in range(K+1):
            # For each count of current letter to use
            for k in range(C[i]+1):
                if j + k <= K:
                    dp[i+1][j+k] = (dp[i+1][j+k] + dp[i][j]) % MOD
    
    # Sum all lengths from 1 to K
    ans = 0
    for i in range(1, K+1):
        ans = (ans + dp[26][i]) % MOD
        
    print(ans)

solve()
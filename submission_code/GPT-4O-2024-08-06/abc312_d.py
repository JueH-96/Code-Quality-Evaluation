# YOUR CODE HERE
def count_valid_parenthesis_strings(S):
    MOD = 998244353
    n = len(S)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1  # Base case: empty string with balance 0
    
    for i in range(1, n + 1):
        for j in range(n + 1):
            if S[i - 1] == '(':
                if j > 0:
                    dp[i][j] = dp[i - 1][j - 1]
            elif S[i - 1] == ')':
                if j < n:
                    dp[i][j] = dp[i - 1][j + 1]
            else:  # S[i - 1] == '?'
                if j > 0:
                    dp[i][j] += dp[i - 1][j - 1]
                if j < n:
                    dp[i][j] += dp[i - 1][j + 1]
            
            dp[i][j] %= MOD
    
    return dp[n][0]

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(count_valid_parenthesis_strings(S))
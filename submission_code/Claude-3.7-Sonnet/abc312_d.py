# YOUR CODE HERE
def count_valid_parenthesis(s):
    MOD = 998244353
    n = len(s)
    
    # dp[j] = number of ways to form a valid prefix with a balance of j
    # (where balance = count of '(' minus count of ')')
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty string has balance 0
    
    for i in range(1, n + 1):
        new_dp = [0] * (n + 1)
        for j in range(min(i + 1, n + 1)):  # Balance can't exceed current length
            if s[i-1] == '(' or s[i-1] == '?':
                if j > 0:  # Adding '(' increases balance by 1
                    new_dp[j] = (new_dp[j] + dp[j-1]) % MOD
            
            if s[i-1] == ')' or s[i-1] == '?':
                # Adding ')' decreases balance by 1 (j+1 is the previous balance)
                # We only add if j+1 is within bounds
                if j+1 <= n:
                    new_dp[j] = (new_dp[j] + dp[j+1]) % MOD
        
        dp = new_dp
    
    return dp[0]  # Answer is ways to get balance 0 after processing the whole string

if __name__ == "__main__":
    s = input().strip()
    print(count_valid_parenthesis(s))
# YOUR CODE HERE
MOD = 998244353

s = input().strip()
n = len(s)

# dp[i][j] = number of ways to process first i characters with balance j
# balance = number of '(' minus number of ')'
dp = {}

def solve(pos, balance):
    if balance < 0:
        return 0
    
    if pos == n:
        return 1 if balance == 0 else 0
    
    if (pos, balance) in dp:
        return dp[(pos, balance)]
    
    result = 0
    
    if s[pos] == '(':
        result = solve(pos + 1, balance + 1)
    elif s[pos] == ')':
        result = solve(pos + 1, balance - 1)
    else:  # s[pos] == '?'
        # Try both '(' and ')'
        result = (solve(pos + 1, balance + 1) + solve(pos + 1, balance - 1)) % MOD
    
    dp[(pos, balance)] = result
    return result

print(solve(0, 0))
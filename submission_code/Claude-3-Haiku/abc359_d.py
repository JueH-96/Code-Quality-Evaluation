import math

def is_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_good_strings(s, k):
    n = len(s)
    q = s.count('?')
    dp = [[0] * (1 << q) for _ in range(n+1)]
    dp[0][0] = 1
    
    for i in range(1, n+1):
        for mask in range(1 << q):
            if s[i-1] == 'A':
                dp[i][mask] = dp[i-1][mask]
            elif s[i-1] == 'B':
                dp[i][mask] = dp[i-1][mask]
            else:
                dp[i][mask] = dp[i-1][mask] * 2
            
            if i >= k and not is_palindrome(s[:i], k):
                for j in range(q):
                    if (mask >> j) & 1:
                        dp[i][(mask ^ (1 << j))] = (dp[i][(mask ^ (1 << j))] + dp[i-1][mask]) % 998244353
    
    return sum(dp[n]) % 998244353

# Read input
n, k = map(int, input().split())
s = input()

# Solve the problem
result = count_good_strings(s, k)

# Print the answer
print(result)
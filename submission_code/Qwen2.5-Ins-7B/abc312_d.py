# YOUR CODE HERE
from collections import deque

def solve(S):
    MOD = 998244353
    n = len(S)
    dp = [0] * (n + 1)
    dp[0] = 1
    stack = deque()
    
    for i, c in enumerate(S):
        if c == '(':
            if stack and stack[-1] == '?':
                dp[i + 1] = (dp[i + 1] + dp[i - 1]) % MOD
                stack.pop()
            else:
                stack.append('(')
        elif c == ')':
            if stack and stack[-1] == '(':
                dp[i + 1] = (dp[i + 1] + dp[i - 1]) % MOD
                stack.pop()
            else:
                stack.append(')')
        elif c == '?':
            if stack and stack[-1] == '(':
                dp[i + 1] = (dp[i + 1] + dp[i - 1]) % MOD
                stack.pop()
            else:
                stack.append('?')
    
    for i in range(n - 1, -1, -1):
        if S[i] == '(':
            if stack and stack[-1] == '?':
                dp[i] = (dp[i] + dp[i + 2]) % MOD
                stack.pop()
            else:
                stack.append('(')
        elif S[i] == ')':
            if stack and stack[-1] == '(':
                dp[i] = (dp[i] + dp[i + 2]) % MOD
                stack.pop()
            else:
                stack.append(')')
        elif S[i] == '?':
            if stack and stack[-1] == '(':
                dp[i] = (dp[i] + dp[i + 2]) % MOD
                stack.pop()
            else:
                stack.append('?')
    
    return dp[0]

if __name__ == "__main__":
    S = input().strip()
    print(solve(S))
# YOUR CODE HERE
MOD = 998244353

s = input()
q_count = s.count('?')
dp = {}

def count_valid(string):
    if string in dp:
        return dp[string]
    
    count = 0
    stack = []
    for char in string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                dp[string] = 0
                return 0
            stack.pop()
    
    if not stack:
        dp[string] = 1
        return 1
    else:
        dp[string] = 0
        return 0

def solve(index, balance, q_left):
    if index == len(s):
        return 1 if balance == 0 else 0
    
    if (index, balance, q_left) in dp:
        return dp[(index, balance, q_left)]
    
    char = s[index]
    ans = 0
    if char == '(':
        ans = solve(index + 1, balance + 1, q_left)
    elif char == ')':
        ans = solve(index + 1, balance - 1, q_left) if balance > 0 else 0
    else:
        ans = (solve(index + 1, balance + 1, q_left - 1) + solve(index + 1, balance - 1, q_left - 1)) % MOD
    
    dp[(index, balance, q_left)] = ans
    return ans

print(solve(0, 0, q_count))
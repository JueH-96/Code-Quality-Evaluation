# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def count_parentheses(s):
    n = len(s)
    dp = defaultdict(int)
    dp[0, 0] = 1

    for i in range(n):
        new_dp = defaultdict(int)
        for balance, count in dp.items():
            if s[i] == '(' or s[i] == '?':
                new_dp[balance + 1] = (new_dp[balance + 1] + count) % MOD
            if s[i] == ')' or s[i] == '?':
                if balance > 0:
                    new_dp[balance - 1] = (new_dp[balance - 1] + count) % MOD
        dp = new_dp

    return dp[0]

s = input().strip()
print(count_parentheses(s))
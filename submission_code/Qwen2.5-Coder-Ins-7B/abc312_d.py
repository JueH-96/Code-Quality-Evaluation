# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]

MOD = 998244353

@lru_cache(None)
def dp(s):
    if s == "":
        return 1
    if s[0] == "(" and s[-1] == ")":
        return dp(s[1:-1])
    if s[0] == "(":
        return dp(s[1:])
    if s[-1] == ")":
        return dp(s[:-1])
    if s[0] == "?" and s[-1] == "?":
        return (dp(s[1:-1]) * 2) % MOD
    if s[0] == "?":
        return dp(s[1:]) % MOD
    if s[-1] == "?":
        return dp(s[:-1]) % MOD
    return 0

print(dp(S) % MOD)
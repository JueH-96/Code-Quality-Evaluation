from functools import lru_cache

def count_parenthesis_strings(s):
    n = len(s)
    mod = 998244353

    @lru_cache(maxsize=None)
    def dp(i):
        if i == n:
            return 1
        if s[i] == '(':
            return dp(i + 1)
        if s[i] == ')':
            return dp(i + 1)
        res = 0
        res = (res + dp(i + 1)) % mod
        res = (res + dp(i + 1)) % mod
        return res

    return sum(dp(i) for i in range(n)) % mod

if __name__ == '__main__':
    s = input()
    print(count_parenthesis_strings(s))
import sys
from collections import defaultdict

MOD = 998244353

def count_non_ddos(s):
    n = len(s)
    q = s.count('?')
    dp = defaultdict(int)
    dp[('', 0)] = 1

    for i in range(n):
        if s[i] == '?':
            for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                new_s = s[:i] + c + s[i+1:]
                for prev_s, prev_q in dp.keys():
                    if prev_q < q:
                        if (new_s[i-1:i+3] == new_s[i-1:i+3].upper() or
                            new_s[i-1:i+3] == new_s[i-1] + new_s[i-1].lower() + new_s[i+1]):
                            continue
                        new_q = prev_q + (s[i] != c)
                        new_key = (prev_s + new_s[i], new_q)
                        dp[new_key] = (dp[new_key] + dp[(prev_s, prev_q)]) % MOD
        else:
            for prev_s, prev_q in dp.keys():
                if prev_q < q:
                    if (s[i-1:i+3] == s[i-1:i+3].upper() or
                        s[i-1:i+3] == s[i-1] + s[i-1].lower() + s[i+1]):
                        continue
                    new_q = prev_q + (s[i] != s[i-1])
                    new_key = (prev_s + s[i], new_q)
                    dp[new_key] = (dp[new_key] + dp[(prev_s, prev_q)]) % MOD

    return sum(dp.values()) % MOD

if __name__ == '__main__':
    s = input().strip()
    print(count_non_ddos(s))
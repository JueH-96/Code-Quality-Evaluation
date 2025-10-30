#!/usr/bin/env python3
import sys
import threading

def main():
    import sys
    S = sys.stdin.readline().strip()
    mod = 998244353
    n = len(S)
    # dp[j] = number of ways to have balance j after processing some prefix
    dp = [0] * (n+2)
    dp[0] = 1
    for ch in S:
        # next row
        ndp = [0] * (n+2)
        if ch == '(':
            # only '(' increases balance
            for j in range(n):
                if dp[j]:
                    ndp[j+1] = (ndp[j+1] + dp[j]) % mod
        elif ch == ')':
            # only ')' decreases balance
            for j in range(1, n+1):
                if dp[j]:
                    ndp[j-1] = (ndp[j-1] + dp[j]) % mod
        else:  # ch == '?'
            # treat as '('
            for j in range(n):
                if dp[j]:
                    ndp[j+1] = (ndp[j+1] + dp[j]) % mod
            # treat as ')'
            for j in range(1, n+1):
                if dp[j]:
                    ndp[j-1] = (ndp[j-1] + dp[j]) % mod
        dp = ndp
    # the answer is number of ways ending with balance zero
    print(dp[0] % mod)

if __name__ == "__main__":
    main()
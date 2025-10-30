# YOUR CODE HERE
import sys

def solve():
    MOD = 998244353
    S = sys.stdin.read().strip()
    n = len(S)
    
    dp_current = [0] * (n + 2)
    dp_current[0] = 1
    
    for c in S:
        dp_next = [0] * (n + 2)
        for b in range(n + 1):
            if dp_current[b] == 0:
                continue
            if c == '(':
                dp_next[b + 1] = (dp_next[b + 1] + dp_current[b]) % MOD
            elif c == ')':
                if b > 0:
                    dp_next[b - 1] = (dp_next[b - 1] + dp_current[b]) % MOD
            elif c == '?':
                # Replace with '('
                dp_next[b + 1] = (dp_next[b + 1] + dp_current[b]) % MOD
                # Replace with ')'
                if b > 0:
                    dp_next[b - 1] = (dp_next[b - 1] + dp_current[b]) % MOD
        dp_current = dp_next
    
    print(dp_current[0] % MOD)
import sys
import numpy as np

def main():
    mod = 998244353
    s = sys.stdin.readline().strip()
    n = len(s)
    # dp[j] = number of valid ways to achieve a balance j after processing some prefix of s.
    # Because a balanced string must have nonnegative balance at every position and finish with balance 0.
    dp = np.zeros(n+1, dtype=np.int64)
    dp[0] = 1

    # For each character, update the dp vector.
    # Note: When processing i-th character (0-indexed), only dp[0]..dp[i] could be nonzero.
    for i, ch in enumerate(s):
        newdp = np.zeros(n+1, dtype=np.int64)
        if ch == '(':
            # If the character is '(', it increases the current balance by one.
            # For each j in 0..i, newdp[j+1] becomes dp[j].
            newdp[1:i+2] = dp[:i+1]
        elif ch == ')':
            # If the character is ')', it decreases the balance by one.
            # This is only possible if the current balance is at least 1.
            # For each j in 1..i, newdp[j-1] becomes dp[j].
            newdp[0:i] = dp[1:i+1]
        else:  # ch == '?'
            # A '?' can be replaced by either '(' or ')'.
            # Option 1: treat '?' as '(' → increase balance by one.
            newdp[1:i+2] = dp[:i+1]
            # Option 2: treat '?' as ')' → decrease balance by one (only possible if balance >= 1).
            newdp[:i] += dp[1:i+1]
        newdp %= mod
        dp = newdp

    # The answer is the number of ways to have balance 0 after processing all characters.
    sys.stdout.write(str(int(dp[0] % mod)))
    
if __name__ == '__main__':
    main()
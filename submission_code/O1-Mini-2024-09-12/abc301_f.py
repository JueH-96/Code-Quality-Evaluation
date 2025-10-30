# YOUR CODE HERE
import sys

def solve():
    import sys
    MOD = 998244353
    S = sys.stdin.read().strip()
    n = len(S)
    dp0 = 1
    dp1 = [0] * 26
    dp2 = [0] * 26
    dp3 = [0] * 26
    for ch in S:
        # Determine U_chars and L_count
        if ch == '?':
            U_chars = list(range(26))
            U_count = 26
            L_count = 26
        elif 'A' <= ch <= 'Z':
            U_chars = [ord(ch) - ord('A')]
            U_count = 1
            L_count = 0
        elif 'a' <= ch <= 'z':
            U_chars = []
            U_count = 0
            L_count = 1
        else:
            U_chars = []
            U_count = 0
            L_count = 0
        # Initialize new dp variables
        new_dp0 = 0
        new_dp1 = [0] * 26
        new_dp2 = [0] * 26
        new_dp3 = [0] * 26
        # Handle lowercase contributions
        if L_count > 0:
            new_dp0 = (new_dp0 + dp0 * L_count) % MOD
            for c in range(26):
                new_dp1[c] = (new_dp1[c] + dp1[c] * L_count) % MOD
                new_dp2[c] = (new_dp2[c] + dp2[c] * L_count) % MOD
                new_dp3[c] = (new_dp3[c] + dp3[c] * L_count + dp2[c] * L_count) % MOD
        # Handle uppercase contributions
        if U_count > 0:
            for c in U_chars:
                # Update dp1[c]
                new_dp1[c] = (new_dp1[c] + dp0) % MOD
                # Update dp2[c]
                new_dp2[c] = (new_dp2[c] + dp1[c]) % MOD
                # Update dp3[c]
                new_dp3[c] = (new_dp3[c] + dp2[c]) % MOD
        # Update dp0, dp1, dp2, dp3
        dp0, dp1, dp2, dp3 = new_dp0, new_dp1, new_dp2, new_dp3
    # Calculate the final answer
    total = dp0
    for c in range(26):
        total = (total + dp1[c] + dp2[c] + dp3[c]) % MOD
    print(total)
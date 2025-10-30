# YOUR CODE HERE
MOD = 998244353

def count_non_ddos_strings(S):
    n = len(S)
    dp = [[[0] * 2 for _ in range(2)] for _ in range(2)]
    dp[0][0][0] = 1

    for char in S:
        if char == '?':
            new_dp = [[[0] * 2 for _ in range(2)] for _ in range(2)]
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        # Case 1: Add an uppercase letter
                        new_dp[a][b][c] = (new_dp[a][b][c] + dp[a][b][c] * 26) % MOD
                        # Case 2: Add a lowercase letter
                        new_dp[a][b][c] = (new_dp[a][b][c] + dp[a][b][c] * 26) % MOD
                        # Case 3: Add the same uppercase letter as the first one
                        if a == 0:
                            new_dp[1][b][c] = (new_dp[1][b][c] + dp[a][b][c] * 26) % MOD
                        # Case 4: Add the same uppercase letter as the second one
                        if b == 0:
                            new_dp[a][1][c] = (new_dp[a][1][c] + dp[a][b][c] * 26) % MOD
                        # Case 5: Add a lowercase letter as the third one
                        if c == 0:
                            new_dp[a][b][1] = (new_dp[a][b][1] + dp[a][b][c] * 26) % MOD
            dp = new_dp
        else:
            new_dp = [[[0] * 2 for _ in range(2)] for _ in range(2)]
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        if char.isupper():
                            # Case 1: Add an uppercase letter
                            new_dp[a][b][c] = (new_dp[a][b][c] + dp[a][b][c]) % MOD
                            # Case 2: Add the same uppercase letter as the first one
                            if a == 0 and char == S[0]:
                                new_dp[1][b][c] = (new_dp[1][b][c] + dp[a][b][c]) % MOD
                            # Case 3: Add the same uppercase letter as the second one
                            if b == 0 and char == S[1]:
                                new_dp[a][1][c] = (new_dp[a][1][c] + dp[a][b][c]) % MOD
                        else:
                            # Case 4: Add a lowercase letter
                            new_dp[a][b][c] = (new_dp[a][b][c] + dp[a][b][c]) % MOD
                            # Case 5: Add a lowercase letter as the third one
                            if c == 0:
                                new_dp[a][b][1] = (new_dp[a][b][1] + dp[a][b][c]) % MOD
            dp = new_dp

    result = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                result = (result + dp[a][b][c]) % MOD

    return result

import sys
S = sys.stdin.read().strip()
print(count_non_ddos_strings(S))
# YOUR CODE HERE
import sys

# Read input string S
S = sys.stdin.readline().strip()
N = len(S)
MOD = 998244353

# DP table: dp[i][bal] = number of ways to fill the prefix S[0...i-1]
# such that the resulting substring has a balance of 'bal' (number of '(' - number of ')')
# and the balance has never dropped below zero during the scan of S[0...i-1].
# i: index in the string S (from 0 to N)
# bal: current balance (from 0 to N)
dp = [[0] * (N + 1) for _ in range(N + 1)]

# Base case: Before processing any character (at index 0), the prefix S[0...-1] (empty string)
# has a balance of 0. There is one way to achieve this (the empty way).
dp[0][0] = 1

# Iterate through the string characters from left to right
for i in range(N):
    # Iterate through all possible current balances achieved after processing S[0...i-1]
    for bal in range(N + 1):
        # If there are no ways to reach this state (i, bal), skip it
        if dp[i][bal] == 0:
            continue

        # Consider the character S[i]
        char = S[i]

        # If S[i] is '(' or '?': It can contribute an opening bracket.
        if char == '(' or char == '?':
            # Adding '(' increases the balance by 1.
            new_bal = bal + 1
            # The new balance must be within the valid range [0, N].
            # Maximum possible balance is N (e.g., "((...(").
            if new_bal <= N:
                # Add the number of ways to reach the current state (i, bal)
                # to the number of ways to reach the next state (i+1, new_bal).
                dp[i + 1][new_bal] = (dp[i + 1][new_bal] + dp[i][bal]) % MOD

        # If S[i] is ')' or '?': It can contribute a closing bracket.
        if char == ')' or char == '?':
            # Adding ')' decreases the balance by 1.
            new_bal = bal - 1
            # The new balance must be non-negative to satisfy the prefix balance condition.
            # This implies the current balance must be greater than 0 to become non-negative (0) or positive.
            if new_bal >= 0: # Equivalent to bal > 0, as bal is always >= 0 here
                # Add the number of ways to reach the current state (i, bal)
                # to the number of ways to reach the next state (i+1, new_bal).
                dp[i + 1][new_bal] = (dp[i + 1][new_bal] + dp[i][bal]) % MOD

# After processing the entire string S (up to index N-1), the total number of ways
# to form a valid parenthesis string is the number of ways to finish with a balance of 0.
# This is stored in dp[N][0].
print(dp[N][0])
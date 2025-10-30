# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.readline().strip()

N = len(S)

# dp[i] will store the minimum number of button presses 
# to display the prefix S[0...i-1] (which has length i).
# The array size is N+1 to store dp[0] through dp[N].
dp = [0] * (N + 1)

# Base case: dp[0] = 0. This means an empty string takes 0 presses.
# Python's list initialization `[0] * (N + 1)` already sets dp[0] to 0.

# Iterate from i = 1 to N. dp[i] corresponds to prefix S[:i].
# S[:i] is S[0], S[1], ..., S[i-1].
for i in range(1, N + 1):
    # Option 1: Append the single character S[i-1].
    # This transition comes from the state S[:i-1] (prefix of length i-1), 
    # which took dp[i-1] presses. Adding one more press for S[i-1].
    # This is always a possibility to reach state i.
    dp[i] = dp[i-1] + 1
    
    # Option 2: Append "00".
    # This is possible only if i >= 2 (so S[:i] has at least two characters S[i-2] and S[i-1])
    # AND these last two characters are '0' and '0'.
    if i >= 2:
        # S[i-1] is the i-th character of S (0-indexed) and is the last character of prefix S[:i].
        # S[i-2] is the (i-1)-th character of S and is the second to last character of S[:i].
        if S[i-2] == '0' and S[i-1] == '0':
            # This transition comes from state S[:i-2] (prefix of length i-2),
            # which took dp[i-2] presses. Adding one more press for the "00" button.
            # We take the minimum if this path is better than the one from Option 1.
            dp[i] = min(dp[i], dp[i-2] + 1)
            
# The final answer is dp[N], which is the minimum presses for the full string S.
# Print output to stdout
sys.stdout.write(str(dp[N]) + "
")
# YOUR CODE ENDS HERE
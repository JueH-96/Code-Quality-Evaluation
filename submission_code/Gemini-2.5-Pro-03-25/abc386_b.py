# YOUR CODE HERE
import sys

def solve():
    """
    Solves the calculator button press problem using dynamic programming
    with O(1) space complexity. Reads the input string S from stdin
    and prints the minimum number of button presses to stdout.
    """
    # Read input string from standard input
    s = sys.stdin.readline().strip()
    n = len(s)

    # Constraints state n >= 1, so we don't need to handle n=0 explicitly.

    # Initialize dp variables for O(1) space DP.
    # dp[k] represents the minimum number of button presses to display the prefix S[:k].
    # We only need the values for dp[i-1] and dp[i-2] to calculate dp[i].

    # dp_prev2 stores the value of dp[i-2] relative to the current iteration i.
    # dp_prev1 stores the value of dp[i-1] relative to the current iteration i.
    # Initialize based on dp[0] = 0 (0 presses for an empty string).
    # Before the first iteration (i=1):
    #   dp_prev1 should represent dp[0] = 0.
    #   dp_prev2 represents dp[-1], which can be considered 0 for the logic to work when i=2.
    dp_prev2 = 0
    dp_prev1 = 0

    # Iterate through the string S, calculating dp[i] for each prefix S[:i]
    # The loop variable `i` represents the length of the prefix being considered (from 1 to n).
    for i in range(1, n + 1):
        # Calculate `dp_curr`, which will hold the value of dp[i].

        # Option 1: Assume the last button pressed was the single digit S[i-1].
        # The number of presses is the minimum presses needed for the prefix S[:i-1] (stored in dp_prev1)
        # plus 1 press for the current digit S[i-1].
        dp_curr = dp_prev1 + 1

        # Option 2: Check if the last button pressed could have been '00'.
        # This is possible only if the current prefix length `i` is at least 2,
        # and the last two characters of the prefix S[:i] (which are S[i-2] and S[i-1]) form "00".
        # We access these characters using the slice s[i-2:i].
        if i >= 2 and s[i-2:i] == "00":
            # If appending "00" is possible, the number of presses is the minimum presses needed
            # for the prefix S[:i-2] (stored in dp_prev2) plus 1 press for the '00' button.
            # We choose the minimum cost between Option 1 (appending single digit) and Option 2 (appending "00").
            dp_curr = min(dp_curr, dp_prev2 + 1)

        # Prepare the dp variables for the next iteration (where the prefix length will be i+1).
        # The value that was dp[i-1] (stored in dp_prev1) now becomes the value for dp[(i+1)-2].
        # The value that was dp[i] (stored in dp_curr) now becomes the value for dp[(i+1)-1].
        dp_prev2 = dp_prev1
        dp_prev1 = dp_curr

    # After the loop completes, `i` has reached `n`.
    # `dp_prev1` now holds the value of dp[n], which is the minimum number of button presses
    # required to display the entire string S.
    print(dp_prev1)

# Execute the solver function
solve()
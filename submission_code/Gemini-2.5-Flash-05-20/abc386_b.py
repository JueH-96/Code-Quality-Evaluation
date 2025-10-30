import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)

    # dp[i] will store the minimum number of button presses
    # to display the prefix S[0...i-1] (string of length i)
    # Initialize with a value representing infinity
    dp = [float('inf')] * (N + 1)

    # Base case: 0 presses for an empty string
    dp[0] = 0

    # Iterate from i = 1 to N to fill the dp array
    for i in range(1, N + 1):
        # Option 1: Append a single digit S[i-1]
        # This means we form S[0...i-1] by taking the minimum presses
        # for S[0...i-2] (dp[i-1]) and adding 1 for the current digit button press.
        dp[i] = dp[i-1] + 1

        # Option 2: Append "00"
        # This is possible if:
        #   a) The current prefix length 'i' is at least 2 (to accommodate "00")
        #   b) The last two characters of the prefix S[0...i-1] are "00"
        # If these conditions are met, we can potentially form S[0...i-1] by
        # taking the minimum presses for S[0...i-3] (dp[i-2]) and adding 1
        # for the "00" button press.
        if i >= 2 and S[i-2:i] == "00":
            dp[i] = min(dp[i], dp[i-2] + 1)

    # The minimum presses to display the entire string S is dp[N]
    print(dp[N])

# Call the solve function to run the program
solve()
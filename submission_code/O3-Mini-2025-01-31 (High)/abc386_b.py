def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # dp[i] will hold the minimum button presses to produce the first i characters of S.
    dp = [10**9] * (n + 1)
    dp[0] = 0
    
    # We have two options at any given position:
    # 1. Press a button corresponding to a single digit (0-9).
    # 2. Press the "00" button (if the next two characters in S are "00").
    #
    # We use dynamic programming to update the minimum number of presses needed to produce S up to each index.
    for i in range(n):
        # Option 1: Add a single digit.
        if i + 1 <= n:
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        # Option 2: Add "00" if the next two characters are exactly "00".
        if i + 2 <= n and S[i:i+2] == "00":
            dp[i + 2] = min(dp[i + 2], dp[i] + 1)
    
    sys.stdout.write(str(dp[n]))

if __name__ == '__main__':
    main()
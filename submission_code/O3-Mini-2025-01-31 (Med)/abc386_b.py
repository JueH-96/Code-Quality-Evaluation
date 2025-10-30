def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    S = data[0]
    n = len(S)
    
    # dp[i] represents the minimum number of button presses to build S[:i]
    # Initialize dp[0] = 0 (empty string) and high value for the rest.
    dp = [10**9] * (n + 1)
    dp[0] = 0
    
    # The available buttons: "00", and each single digit "0" to "9"
    buttons = ["00"] + [str(i) for i in range(10)]
    
    # Compute dp using a simple dynamic programming approach.
    for i in range(1, n + 1):
        for button in buttons:
            if i >= len(button) and S[i - len(button):i] == button:
                dp[i] = min(dp[i], dp[i - len(button)] + 1)
    
    sys.stdout.write(str(dp[n]))
    
if __name__ == '__main__':
    main()
def main():
    import sys
    S = sys.stdin.readline().strip()
    N = len(S)
    # dp[i] = minimum button presses to form S[:i]
    dp = [10**9] * (N + 1)
    dp[0] = 0

    for i in range(1, N + 1):
        # press one digit (always possible)
        dp[i] = dp[i - 1] + 1
        # if the last two characters are "00", we can press the "00" button
        if i >= 2 and S[i - 2:i] == "00":
            dp[i] = min(dp[i], dp[i - 2] + 1)

    print(dp[N])

if __name__ == "__main__":
    main()
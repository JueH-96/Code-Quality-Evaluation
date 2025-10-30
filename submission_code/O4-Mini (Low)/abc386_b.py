def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    # dp[i] = minimum presses to build S[:i]
    # dp[0] = 0 (empty string)
    dp = [0] * (n + 1)
    INF = 10**9
    for i in range(1, n + 1):
        # press single digit S[i-1]
        dp[i] = dp[i - 1] + 1
        # if last two characters are "00", we can press the "00" button
        if i >= 2 and S[i-2:i] == "00":
            dp[i] = min(dp[i], dp[i-2] + 1)
    print(dp[n])

if __name__ == "__main__":
    main()
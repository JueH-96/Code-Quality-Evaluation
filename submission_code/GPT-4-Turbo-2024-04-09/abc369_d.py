def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][0] - max exp points using first i monsters with i-th monster not defeated
    # dp[i][1] - max exp points using first i monsters with i-th monster defeated as odd-numbered defeat
    # dp[i][2] - max exp points using first i monsters with i-th monster defeated as even-numbered defeat
    dp = [[0] * 3 for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        strength = A[i - 1]
        dp[i][0] = max(dp[i - 1])  # Letting go of the i-th monster
        dp[i][1] = dp[i - 1][0] + strength  # Defeating i-th as odd-numbered (after a let go)
        if i > 1:
            dp[i][1] = max(dp[i][1], dp[i - 1][2] + strength)  # Defeating i-th as odd-numbered (after an even-numbered)
            dp[i][2] = dp[i - 1][1] + 2 * strength  # Defeating i-th as even-numbered (after an odd-numbered)
    
    # The answer is the maximum of the last row of dp table
    print(max(dp[N]))

if __name__ == "__main__":
    main()
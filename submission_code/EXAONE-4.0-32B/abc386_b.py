def main():
    S = input().strip()
    n = len(S)
    dp = [0] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        dp[i] = dp[i-1] + 1
        if i >= 2 and S[i-2:i] == "00":
            dp[i] = min(dp[i], dp[i-2] + 1)
    
    print(dp[n])

if __name__ == '__main__':
    main()
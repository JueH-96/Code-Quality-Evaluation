def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    n = len(S)
    
    # dp[i] will be the minimum button presses needed to form S[:i]
    dp = [10**9] * (n + 1)
    dp[0] = 0
    
    for i in range(n):
        # Option 1: Press one-digit button
        if i + 1 <= n:
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        # Option 2: Press button "00" if possible.
        if i + 2 <= n and S[i:i+2] == "00":
            dp[i + 2] = min(dp[i + 2], dp[i] + 1)
    # Print answer corresponding to full string length
    print(dp[n])

if __name__ == '__main__':
    main()
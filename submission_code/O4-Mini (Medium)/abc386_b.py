import sys

def main():
    S = sys.stdin.readline().strip()
    n = len(S)
    # dp[i]: minimum presses to build S[:i]
    INF = n + 5
    dp = [INF] * (n + 1)
    dp[0] = 0

    for i in range(n):
        # Press a single digit button (0-9)
        # Always allowed: matches S[i]
        dp[i+1] = min(dp[i+1], dp[i] + 1)

        # Press "00" if the next two chars are "00"
        if i + 2 <= n and S[i] == '0' and S[i+1] == '0':
            dp[i+2] = min(dp[i+2], dp[i] + 1)

    print(dp[n])

if __name__ == "__main__":
    main()
import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    if n == 1:
        print(0)
        return
    # dp[i] is the maximum score for the first i elements
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + abs(a[i - 1] - a[i - 2]))
    print(dp[n])

if __name__ == "__main__":
    main()
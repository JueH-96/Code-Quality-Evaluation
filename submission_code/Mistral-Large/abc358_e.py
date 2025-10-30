import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    K = int(data[0])
    C = list(map(int, data[1:]))

    # Initialize dp array
    dp = [[0] * (K + 1) for _ in range(27)]
    dp[0][0] = 1

    # Fill dp array
    for i in range(1, 27):
        for j in range(K + 1):
            dp[i][j] = dp[i - 1][j]
            if j > 0:
                dp[i][j] += dp[i][j - 1]
                dp[i][j] += dp[i - 1][j - 1] * (C[i - 1])
                dp[i][j] %= MOD

    result = 0
    for j in range(1, K + 1):
        result += dp[26][j]
        result %= MOD

    print(result)

if __name__ == "__main__":
    main()
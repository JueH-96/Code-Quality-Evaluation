MOD = 998244353

def count_valid_operations(N, Q, P, V):
    dp = [[0] * (N + 1) for _ in range(Q + 1)]
    dp[0][0] = 1

    for i in range(1, Q + 1):
        p, v = P[i - 1], V[i - 1]
        for j in range(N + 1):
            if j < p and dp[i - 1][j] > 0:
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD
            if j >= p and dp[i - 1][j] > 0 and j < N and v <= dp[i - 1][j]:
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD

    return sum(dp[Q]) % MOD

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    Q = int(data[1])
    P = [int(data[2 * i + 2]) for i in range(Q)]
    V = [int(data[2 * i + 3]) for i in range(Q)]

    result = count_valid_operations(N, Q, P, V)
    print(result)

if __name__ == "__main__":
    main()
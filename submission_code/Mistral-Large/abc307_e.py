import sys

MOD = 998244353

def solve(N, M):
    if N == 1:
        return M
    if M == 1:
        return 0

    # dp[i][j] will store the number of ways to assign colors to the first i people
    # such that the ith person gets color j and no two adjacent people have the same color.
    dp = [[0] * M for _ in range N]

    # Base case: There are M-1 ways to color the first person if the second person is already colored.
    for j in range(M):
        dp[0][j] = 1

    # Fill the dp table
    for i in range(1, N):
        for j in range(M):
            for k in range(M):
                if j != k:
                    dp[i][j] = (dp[i][j] + dp[i-1][k]) % MOD

    # Sum up all the valid configurations for the Nth person
    result = sum(dp[N-1]) % MOD

    # Subtract the cases where the first and the last person have the same color
    for j in range(M):
        result = (result - dp[N-1][j] + MOD) % MOD

    return result

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    print(solve(N, M))

if __name__ == "__main__":
    main()
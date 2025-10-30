import sys
from math import gcd

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Precompute gcd for all pairs
    gcd_matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            gcd_matrix[i][j] = gcd(A[i], A[j])

    # Initialize dp array
    dp = [0] * (N + 1)

    # Compute the sum of scores for all subsequences
    for m in range(1, N + 1):
        for i in range(N - m + 1):
            j = i + m - 1
            dp[m] = (dp[m] + gcd_matrix[i][j]) % MOD
        print(dp[m])

if __name__ == "__main__":
    main()
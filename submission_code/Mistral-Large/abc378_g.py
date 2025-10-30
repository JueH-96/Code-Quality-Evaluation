import sys
from math import comb

def count_permutations(A, B, M):
    AB = A * B
    dp = [[0] * (B + 1) for _ in range(A + 1)]
    dp[0][0] = 1

    for i in range(1, AB):
        new_dp = [[0] * (B + 1) for _ in range(A + 1)]
        for a in range(A + 1):
            for b in range(B + 1):
                if dp[a][b] == 0:
                    continue
                # Add i to the increasing subsequence
                if a + 1 <= A:
                    new_dp[a + 1][b] += dp[a][b]
                    new_dp[a + 1][b] %= M
                # Add i to the decreasing subsequence
                if b + 1 <= B:
                    new_dp[a][b + 1] += dp[a][b]
                    new_dp[a][b + 1] %= M
        dp = new_dp

    # Count valid permutations
    result = 0
    for a in range(1, A + 1):
        for b in range(1, B + 1):
            if a * b == AB:
                result += dp[a][b]
                result %= M

    return result

def main():
    input = sys.stdin.read
    data = input().split()

    A = int(data[0])
    B = int(data[1])
    M = int(data[2])

    result = count_permutations(A, B, M)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()
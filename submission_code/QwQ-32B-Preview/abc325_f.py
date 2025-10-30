import sys

def main():
    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L = [0] * 2
    C = [0] * 2
    K = [0] * 2
    for j in range(2):
        L[j], C[j], K[j] = map(int, sys.stdin.readline().split())

    # Initialize DP
    INF = float('inf')
    dp = [[INF] * (K[0] + 1) for _ in range(K[1] + 1)]
    dp[0][0] = 0

    for d in D:
        new_dp = [[INF] * (K[0] + 1) for _ in range(K[1] + 1)]
        for k1 in range(K[0] + 1):
            for k2 in range(K[1] + 1):
                if dp[k1][k2] == INF:
                    continue
                # Try using only type-1 sensors
                if L[0] > 0:
                    num1 = -(-d // L[0])  # Ceiling division
                    if k1 + num1 <= K[0]:
                        cost1 = num1 * C[0]
                        new_dp[k1 + num1][k2] = min(new_dp[k1 + num1][k2], dp[k1][k2] + cost1)
                # Try using only type-2 sensors
                if L[1] > 0:
                    num2 = -(-d // L[1])  # Ceiling division
                    if k2 + num2 <= K[1]:
                        cost2 = num2 * C[1]
                        new_dp[k1][k2 + num2] = min(new_dp[k1][k2 + num2], dp[k1][k2] + cost2)
                # Try using a combination of type-1 and type-2 sensors
                for x in range(num1 + 1):
                    for y in range(num2 + 1):
                        covered = x * L[0] + y * L[1]
                        if covered >= d:
                            if k1 + x <= K[0] and k2 + y <= K[1]:
                                cost = x * C[0] + y * C[1]
                                new_dp[k1 + x][k2 + y] = min(new_dp[k1 + x][k2 + y], dp[k1][k2] + cost)
        dp = new_dp

    # Find the minimum cost among all possible sensor usage within limits
    min_cost = min(dp[k1][k2] for k1 in range(K[0] + 1) for k2 in range(K[1] + 1))
    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

if __name__ == "__main__":
    main()
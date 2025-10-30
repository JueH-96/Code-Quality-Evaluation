# YOUR CODE HERE
import sys

def max_happiness(N, W, items):
    dp = [[-float('inf')] * (W + 1) for _ in range(N + 1)]
    dp[0][0] = 0

    for i in range(1, N + 1):
        w, v = items[i - 1]
        for j in range(W + 1):
            dp[i][j] = dp[i - 1][j]
            k = 1
            while w * k <= j and k * v - k * k > 0:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - w * k] + k * v - k * k)
                k += 1

    return max(dp[N])

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    W = int(data[1])
    items = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(N)]
    print(max_happiness(N, W, items))
import sys

def solve():
    N, X, Y = map(int, input().split())
    dishes = [tuple(map(int, input().split())) for _ in range(N)]
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    for a, b in dishes:
        for x in range(X, a - 1, -1):
            for y in range(Y, b - 1, -1):
                dp[x][y] = max(dp[x][y], dp[x - a][y - b] + 1)
    print(max(max(row) for row in dp))

if __name__ == "__main__":
    solve()
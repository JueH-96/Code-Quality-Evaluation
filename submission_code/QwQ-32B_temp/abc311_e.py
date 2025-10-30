def main():
    import sys
    H, W, N = map(int, sys.stdin.readline().split())
    grid = [[True for _ in range(W)] for _ in range(H)]
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        grid[a-1][b-1] = False

    prev_dp = [0] * W
    total = 0

    for i in range(H):
        current_dp = [0] * W
        prev_diag = 0
        for j in range(W):
            if not grid[i][j]:
                current_dp[j] = 0
                prev_diag = 0
            else:
                if i == 0 or j == 0:
                    current_dp[j] = 1
                else:
                    val = min(prev_dp[j], current_dp[j-1], prev_diag) + 1
                    current_dp[j] = val
                prev_diag = current_dp[j]
        total += sum(current_dp)
        prev_dp = current_dp

    print(total)

if __name__ == "__main__":
    main()
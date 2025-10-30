def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1

    holes = set()
    for _ in range(N):
        a = int(input[ptr]) - 1
        ptr += 1
        b = int(input[ptr]) - 1
        ptr += 1
        holes.add((a, b))

    dp = [[0] * W for _ in range(H)]

    for i in reversed(range(H)):
        for j in reversed(range(W)):
            if (i, j) in holes:
                dp[i][j] = 0
            else:
                if i == H - 1 or j == W - 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1]) + 1

    total = sum(sum(row) for row in dp)
    print(total)

if __name__ == "__main__":
    main()
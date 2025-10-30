import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W, K = map(int, input().split())
    grid = [input().strip() for _ in range(H)]

    INF = 10 ** 9
    ans = INF

    # Horizontal segments
    if W >= K:
        for i in range(H):
            row = grid[i]
            x_cnt = 0
            dot_cnt = 0

            # first window
            for j in range(K):
                c = row[j]
                if c == 'x':
                    x_cnt += 1
                elif c == '.':
                    dot_cnt += 1
            if x_cnt == 0:
                ans = min(ans, dot_cnt)

            # slide window
            for j in range(K, W):
                out_c = row[j - K]
                if out_c == 'x':
                    x_cnt -= 1
                elif out_c == '.':
                    dot_cnt -= 1

                in_c = row[j]
                if in_c == 'x':
                    x_cnt += 1
                elif in_c == '.':
                    dot_cnt += 1

                if x_cnt == 0:
                    ans = min(ans, dot_cnt)

    # Vertical segments
    if H >= K:
        for j in range(W):
            x_cnt = 0
            dot_cnt = 0

            # first window
            for i in range(K):
                c = grid[i][j]
                if c == 'x':
                    x_cnt += 1
                elif c == '.':
                    dot_cnt += 1
            if x_cnt == 0:
                ans = min(ans, dot_cnt)

            # slide window
            for i in range(K, H):
                out_c = grid[i - K][j]
                if out_c == 'x':
                    x_cnt -= 1
                elif out_c == '.':
                    dot_cnt -= 1

                in_c = grid[i][j]
                if in_c == 'x':
                    x_cnt += 1
                elif in_c == '.':
                    dot_cnt += 1

                if x_cnt == 0:
                    ans = min(ans, dot_cnt)

    print(-1 if ans == INF else ans)

# Call the main function
if __name__ == "__main__":
    main()
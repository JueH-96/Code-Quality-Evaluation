import sys

def main():
    H, W, Q = map(int, input().split())
    grid = [[True] * W for _ in range(H)]

    for _ in range(Q):
        R, C = map(int, input().split())
        R, C = R - 1, C - 1

        if grid[R][C]:
            grid[R][C] = False
        else:
            # Check up
            for i in range(R - 1, -1, -1):
                if grid[i][C]:
                    grid[i][C] = False
                    break

            # Check down
            for i in range(R + 1, H):
                if grid[i][C]:
                    grid[i][C] = False
                    break

            # Check left
            for j in range(C - 1, -1, -1):
                if grid[R][j]:
                    grid[R][j] = False
                    break

            # Check right
            for j in range(C + 1, W):
                if grid[R][j]:
                    grid[R][j] = False
                    break

    count = sum(sum(row) for row in grid)
    print(count)

if __name__ == "__main__":
    main()
def main():
    H, W, Q = map(int, input().split())
    grid = [[True] * W for _ in range(H)]  # True represents a wall
    total_walls = H * W

    for _ in range(Q):
        R, C = map(int, input().split())
        R -= 1  # Convert to 0-based index
        C -= 1

        if grid[R][C]:
            grid[R][C] = False
            total_walls -= 1
        else:
            # Check up
            for i in range(R - 1, -1, -1):
                if grid[i][C]:
                    grid[i][C] = False
                    total_walls -= 1
                    break
            # Check down
            for i in range(R + 1, H):
                if grid[i][C]:
                    grid[i][C] = False
                    total_walls -= 1
                    break
            # Check left
            for j in range(C - 1, -1, -1):
                if grid[R][j]:
                    grid[R][j] = False
                    total_walls -= 1
                    break
            # Check right
            for j in range(C + 1, W):
                if grid[R][j]:
                    grid[R][j] = False
                    total_walls -= 1
                    break

    print(total_walls)

if __name__ == "__main__":
    main()
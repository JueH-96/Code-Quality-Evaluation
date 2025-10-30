import sys

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    grid = [[0] * W for _ in range(H)]
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        grid[a-1][b-1] = 1  # Convert to 0-based indices

    # Precompute right_max
    right_max = [[0]*W for _ in range(H)]
    for i in range(H):
        row = grid[i]
        right_max[i][W-1] = 0 if row[W-1] else 1
        for j in range(W-2, -1, -1):
            if row[j]:
                right_max[i][j] = 0
            else:
                right_max[i][j] = right_max[i][j+1] + 1

    # Precompute down_max
    down_max = [[0]*W for _ in range(H)]
    for j in range(W):
        down_max[H-1][j] = 0 if grid[H-1][j] else 1
        for i in range(H-2, -1, -1):
            if grid[i][j]:
                down_max[i][j] = 0
            else:
                down_max[i][j] = down_max[i+1][j] + 1

    # Compute max_size
    max_size = [[0]*W for _ in range(H)]
    for i in reversed(range(H)):
        for j in reversed(range(W)):
            if grid[i][j]:
                max_size[i][j] = 0
            else:
                next_size = 0
                if i + 1 < H and j + 1 < W:
                    next_size = max_size[i+1][j+1]
                possible_n = 1 + min(right_max[i][j], down_max[i][j], next_size)
                max_size[i][j] = possible_n

    total = sum(sum(row) for row in max_size)
    print(total)

if __name__ == "__main__":
    main()
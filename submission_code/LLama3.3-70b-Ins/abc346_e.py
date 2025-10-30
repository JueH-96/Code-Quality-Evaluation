import sys

def main():
    H, W, M = map(int, sys.stdin.readline().split())
    operations = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Initialize the grid with zeros
    grid = [[0] * W for _ in range(H)]

    # Apply the operations
    for T, A, X in operations:
        if T == 1:
            # Repaint the A-th row with color X
            for j in range(W):
                grid[A-1][j] = X
        else:
            # Repaint the A-th column with color X
            for i in range(H):
                grid[i][A-1] = X

    # Count the occurrences of each color
    color_counts = {}
    for row in grid:
        for color in row:
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1

    # Print the results
    K = len(color_counts)
    print(K)
    for color in sorted(color_counts.keys()):
        print(color, color_counts[color])

if __name__ == "__main__":
    main()
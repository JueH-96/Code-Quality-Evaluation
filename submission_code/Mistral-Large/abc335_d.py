import sys

def generate_grid(N):
    # Initialize the grid with zeros
    grid = [['T' if (i == j == (N + 1) // 2) else 0 for j in range(N)] for i in range(N)]

    # Directions for moving right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_index = 0
    current_value = 1
    i, j = (N + 1) // 2, (N + 1) // 2

    while current_value <= N * N - 1:
        ni, nj = i + directions[direction_index][0], j + directions[direction_index][1]
        if 0 <= ni < N and 0 <= nj < N and grid[ni][nj] == 0:
            grid[ni][nj] = current_value
            current_value += 1
            i, j = ni, nj
        else:
            direction_index = (direction_index + 1) % 4

    return grid

def print_grid(grid):
    for row in grid:
        print(' '.join(map(str, row)))

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    grid = generate_grid(N)
    print_grid(grid)

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = list(map(int, input[1:N+1]))
    Q = list(map(int, input[N+1:2*N+1]))

    # Initialize the grid with all zeros
    grid = [[0] * N for _ in range(N)]

    # Fill the grid according to the permutations P and Q
    for i in range(N):
        grid[P[i] - 1][i] = 1
        grid[i][Q[i] - 1] = 1

    # Ensure the conditions are met by adjusting the grid
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                grid[i][j] = 1
                if not is_valid(grid, P, Q):
                    grid[i][j] = 0

    # Print the grid
    for row in grid:
        print(''.join(map(str, row)))

def is_valid(grid, P, Q):
    N = len(grid)
    # Check rows
    rows = [''.join(map(str, grid[P[i] - 1])) for i in range(N)]
    if rows != sorted(rows):
        return False

    # Check columns
    cols = [''.join(map(str, [grid[j][Q[i] - 1] for j in range(N)])) for i in range(N)]
    if cols != sorted(cols):
        return False

    return True

if __name__ == "__main__":
    main()
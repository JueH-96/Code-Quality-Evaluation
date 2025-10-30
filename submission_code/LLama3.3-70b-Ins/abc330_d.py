import sys

def count_triples(N, grid):
    """
    Count the number of triples of cells that satisfy the conditions.

    Args:
    N (int): The size of the grid.
    grid (list): A list of strings representing the grid.

    Returns:
    int: The number of triples of cells that satisfy the conditions.
    """
    count = 0
    # Iterate over each cell in the grid
    for i in range(N):
        for j in range(N):
            # Check if the cell contains an 'o'
            if grid[i][j] == 'o':
                # Iterate over each other cell in the same row
                for k in range(N):
                    if k != j and grid[i][k] == 'o':
                        # Iterate over each other cell in the same column
                        for row in range(N):
                            if row != i and grid[row][j] == 'o':
                                count += 1
                            if row != i and grid[row][k] == 'o':
                                count += 1
    return count // 2  # Divide by 2 to avoid counting each triple twice

def main():
    # Read the input from stdin
    N = int(sys.stdin.readline().strip())
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Count the number of triples of cells that satisfy the conditions
    count = count_triples(N, grid)

    # Print the answer to stdout
    print(count)

if __name__ == "__main__":
    main()
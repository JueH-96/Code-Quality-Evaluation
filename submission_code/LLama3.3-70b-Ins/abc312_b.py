def is_tak_code(grid, i, j):
    """
    Check if the 9x9 region starting at (i, j) is a TaK Code.

    Args:
    grid (list of lists): The input grid.
    i (int): The row index of the top-left cell.
    j (int): The column index of the top-left cell.

    Returns:
    bool: True if the region is a TaK Code, False otherwise.
    """
    # Check the top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + x][j + y] != '#':
                return False

    # Check the bottom-right 3x3 region
    for x in range(6, 9):
        for y in range(6, 9):
            if grid[i + x][j + y] != '#':
                return False

    # Check the adjacent cells
    for x in range(9):
        for y in range(9):
            if (x < 3 and y < 3) or (x >= 6 and y >= 6):
                continue
            if (x == 3 and y < 3) or (x == 5 and y < 3) or (y == 3 and x < 3) or (y == 5 and x < 3) or \
               (x == 3 and y >= 6) or (x == 5 and y >= 6) or (y == 3 and x >= 6) or (y == 5 and x >= 6):
                if grid[i + x][j + y] != '.':
                    return False

    return True


def find_tak_codes(grid):
    """
    Find all 9x9 regions in the grid that are TaK Codes.

    Args:
    grid (list of lists): The input grid.

    Returns:
    list of tuples: The top-left coordinates of all TaK Code regions.
    """
    tak_codes = []
    for i in range(len(grid) - 8):
        for j in range(len(grid[0]) - 8):
            if is_tak_code(grid, i, j):
                tak_codes.append((i + 1, j + 1))
    return tak_codes


def main():
    n, m = map(int, input().split())
    grid = [list(input()) for _ in range(n)]
    tak_codes = find_tak_codes(grid)
    for code in sorted(tak_codes):
        print(*code)


if __name__ == "__main__":
    main()
def is_tak_code(grid, start_row, start_col):
    """
    Check if the 9x9 region starting at (start_row, start_col) is a valid TaK Code.
    
    Args:
        grid (list of str): The grid representation.
        start_row (int): The starting row of the 9x9 region.
        start_col (int): The starting column of the 9x9 region.
    
    Returns:
        bool: True if the region is a valid TaK Code, False otherwise.
    """
    # Check the top-left 3x3 region (all should be black)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check the bottom-right 3x3 region (all should be black)
    for i in range(6, 9):
        for j in range(6, 9):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check the cells adjacent to the top-left 3x3 region (all should be white)
    # Horizontally adjacent
    for i in range(3):
        if grid[start_row + i][start_col + 3] != '.':
            return False
    
    # Vertically adjacent
    for j in range(3):
        if grid[start_row + 3][start_col + j] != '.':
            return False
    
    # Diagonally adjacent
    if grid[start_row + 3][start_col + 3] != '.':
        return False
    
    # Check the cells adjacent to the bottom-right 3x3 region (all should be white)
    # Horizontally adjacent
    for i in range(6, 9):
        if grid[start_row + i][start_col + 5] != '.':
            return False
    
    # Vertically adjacent
    for j in range(6, 9):
        if grid[start_row + 5][start_col + j] != '.':
            return False
    
    # Diagonally adjacent
    if grid[start_row + 5][start_col + 5] != '.':
        return False
    
    return True

def find_tak_codes(grid, n, m):
    """
    Find all valid TaK Codes in the grid and return their top-left coordinates.
    
    Args:
        grid (list of str): The grid representation.
        n (int): Number of rows in the grid.
        m (int): Number of columns in the grid.
    
    Returns:
        list of tuple: List of top-left coordinates (row, col) of valid TaK Codes.
    """
    result = []
    
    for i in range(n - 8):
        for j in range(m - 8):
            if is_tak_code(grid, i, j):
                result.append((i + 1, j + 1))  # +1 because we're 1-indexed in the output
    
    return result

def main():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    codes = find_tak_codes(grid, n, m)
    
    for row, col in codes:
        print(row, col)

if __name__ == "__main__":
    main()
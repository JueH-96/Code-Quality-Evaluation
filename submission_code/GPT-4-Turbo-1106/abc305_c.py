def find_missing_cookie(grid):
    # Find the rows and columns that have at least one cookie
    rows_with_cookies = [any(cell == '#' for cell in row) for row in grid]
    cols_with_cookies = [any(row[col] == '#' for row in grid) for col in range(len(grid[0]))]

    # Find the topmost and bottommost rows with cookies
    top_row = rows_with_cookies.index(True)
    bottom_row = len(rows_with_cookies) - rows_with_cookies[::-1].index(True) - 1

    # Find the leftmost and rightmost columns with cookies
    left_col = cols_with_cookies.index(True)
    right_col = len(cols_with_cookies) - cols_with_cookies[::-1].index(True) - 1

    # Check for the missing cookie in the identified rectangle
    for i in range(top_row, bottom_row + 1):
        for j in range(left_col, right_col + 1):
            if grid[i][j] == '.':
                return (i + 1, j + 1)

    return None

# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Find and print the missing cookie's position
missing_cookie_position = find_missing_cookie(grid)
print(*missing_cookie_position)
def find_eaten_cookie(grid):
    """
    Find the position of the eaten cookie in the grid.

    Args:
    grid (list of lists): A 2D list representing the grid, where '#' represents a cookie and '.' represents an empty space.

    Returns:
    tuple: The position (i, j) of the eaten cookie.
    """
    # Find the top-left and bottom-right corners of the rectangle
    top_left = None
    bottom_right = None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#':
                if top_left is None:
                    top_left = (i, j)
                bottom_right = (i, j)

    # Check all possible rectangles
    for a in range(top_left[0], bottom_right[0] + 1):
        for b in range(a + 1, bottom_right[0] + 1):
            for c in range(top_left[1], bottom_right[1] + 1):
                for d in range(c + 1, bottom_right[1] + 1):
                    # Check if the current rectangle is valid
                    valid = True
                    for i in range(len(grid)):
                        for j in range(len(grid[0])):
                            if (a <= i <= b and c <= j <= d and grid[i][j] == '.') or \
                               (not (a <= i <= b and c <= j <= d) and grid[i][j] == '#'):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Find the position of the eaten cookie in the current rectangle
                        for i in range(a, b + 1):
                            for j in range(c, d + 1):
                                if grid[i][j] == '.':
                                    return (i + 1, j + 1)

def main():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    i, j = find_eaten_cookie(grid)
    print(i, j)

if __name__ == "__main__":
    main()
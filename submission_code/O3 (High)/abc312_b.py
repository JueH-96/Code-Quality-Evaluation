import sys

def build_relative_positions():
    """
    Pre-compute the relative coordinates (within a 9×9 square)
    that have to be black or white.
    """
    black = []          # mandatory black cells
    white = []          # mandatory white cells

    # top-left 3×3 block (rows 0-2, cols 0-2) → black
    for r in range(3):
        for c in range(3):
            black.append((r, c))

    # bottom-right 3×3 block (rows 6-8, cols 6-8) → black
    for r in range(6, 9):
        for c in range(6, 9):
            black.append((r, c))

    # cells adjacent (4-connected or diagonal) to the top-left block → white
    for r in range(3):        # right side
        white.append((r, 3))
    for c in range(3):        # bottom side
        white.append((3, c))
    white.append((3, 3))      # bottom-right corner

    # cells adjacent to the bottom-right block → white
    for r in range(6, 9):     # left side
        white.append((r, 5))
    for c in range(6, 9):     # top side
        white.append((5, c))
    white.append((5, 5))      # top-left corner

    return black, white


def is_tak_code(top, left, grid, black_cells, white_cells):
    """
    Check whether the 9×9 sub-grid whose top-left corner is (top,left)
    is a valid TaK Code.
    """
    # check blacks
    for dr, dc in black_cells:
        if grid[top + dr][left + dc] != '#':
            return False

    # check whites
    for dr, dc in white_cells:
        if grid[top + dr][left + dc] != '.':
            return False

    return True


def main() -> None:
    N, M = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().rstrip() for _ in range(N)]

    black_cells, white_cells = build_relative_positions()

    answers = []
    for i in range(N - 8):           # top row of 9×9 window
        for j in range(M - 8):       # left column of 9×9 window
            if is_tak_code(i, j, grid, black_cells, white_cells):
                # convert to 1-based indices for output
                answers.append((i + 1, j + 1))

    for r, c in answers:
        print(r, c)


if __name__ == "__main__":
    main()
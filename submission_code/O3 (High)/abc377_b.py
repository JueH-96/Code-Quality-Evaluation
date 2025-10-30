import sys

def main() -> None:
    # read the 8×8 board
    board = [sys.stdin.readline().strip() for _ in range(8)]

    # rows that contain no '#'
    clean_rows = sum(1 for row in board if '#' not in row)

    # columns that contain no '#'
    clean_cols = 0
    for col in range(8):
        if all(board[row][col] == '.' for row in range(8)):
            clean_cols += 1

    # every safe square is at the intersection of a clean row and a clean column
    print(clean_rows * clean_cols)

if __name__ == "__main__":
    main()
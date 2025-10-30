import sys

def is_tak_code(grid, i, j):
    # Check the top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + x][j + y] != '#':
                return False

    # Check the bottom-right 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + 6 + x][j + 6 + y] != '#':
                return False

    # Check the adjacent cells to the top-left 3x3 region
    for x in range(5):
        for y in range(5):
            if 3 <= x + y <= 6 and grid[i - 2 + x][j - 2 + y] != '.':
                return False

    # Check the adjacent cells to the bottom-right 3x3 region
    for x in range(5):
        for y in range(5):
            if 3 <= x + y <= 6 and grid[i + 4 + x][j + 4 + y] != '.':
                return False

    return True

def find_tak_codes(N, M, grid):
    tak_codes = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                tak_codes.append((i + 1, j + 1))
    return tak_codes

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    grid = data[2:]

    tak_codes = find_tak_codes(N, M, grid)

    for code in tak_codes:
        print(code[0], code[1])

if __name__ == "__main__":
    main()
def is_tak_code(grid, start_row, start_col):
    # Check the top-left 3x3 region
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check the bottom-right 3x3 region
    for i in range(3):
        for j in range(3):
            if grid[start_row + 6 + i][start_col + 6 + j] != '#':
                return False
    
    # Check the adjacent cells to the top-left 3x3 region
    for i in range(-1, 5):
        for j in range(-1, 4):
            if 0 <= start_row + i < len(grid) and 0 <= start_col + j < len(grid[0]):
                if (0 <= i < 3 and 0 <= j < 3):
                    continue  # Skip the top-left 3x3
                if (0 <= i < 3 and 6 <= j < 9):
                    continue  # Skip the bottom-right 3x3
                if grid[start_row + i][start_col + j] != '.':
                    return False
    
    # Check the adjacent cells to the bottom-right 3x3 region
    for i in range(6, 9):
        for j in range(6, 9):
            if 0 <= start_row + i < len(grid) and 0 <= start_col + j < len(grid[0]):
                if (6 <= i < 9 and 6 <= j < 9):
                    continue  # Skip the bottom-right 3x3
                if (3 <= i < 6 and 3 <= j < 6):
                    continue  # Skip the top-left 3x3
                if grid[start_row + i][start_col + j] != '.':
                    return False

    return True

def find_tak_codes(N, M, grid):
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                results.append((i + 1, j + 1))  # Convert to 1-based indexing
    return results

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
grid = data[1:N + 1]

results = find_tak_codes(N, M, grid)

for result in results:
    print(result[0], result[1])
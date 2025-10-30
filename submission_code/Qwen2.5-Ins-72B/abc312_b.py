# YOUR CODE HERE
def is_tak_code(grid, i, j):
    # Check top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + x][j + y] != '#':
                return False
    
    # Check bottom-right 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + x + 6][j + y + 6] != '#':
                return False
    
    # Check the 14 cells adjacent to the top-left and bottom-right 3x3 regions
    adjacent_cells = [
        (i, j + 3), (i, j + 4), (i, j + 5), (i + 1, j + 3), (i + 1, j + 5), (i + 2, j + 3), (i + 2, j + 5),
        (i + 3, j), (i + 3, j + 1), (i + 3, j + 2), (i + 3, j + 3), (i + 3, j + 5), (i + 3, j + 6), (i + 3, j + 7),
        (i + 4, j), (i + 4, j + 1), (i + 4, j + 2), (i + 4, j + 3), (i + 4, j + 5), (i + 4, j + 6), (i + 4, j + 7),
        (i + 5, j), (i + 5, j + 1), (i + 5, j + 2), (i + 5, j + 3), (i + 5, j + 5), (i + 5, j + 6), (i + 5, j + 7),
        (i + 6, j + 3), (i + 6, j + 4), (i + 6, j + 5), (i + 7, j + 3), (i + 7, j + 5), (i + 8, j + 3), (i + 8, j + 5)
    ]
    
    for x, y in adjacent_cells:
        if grid[x][y] != '.':
            return False
    
    return True

def find_tak_codes(N, M, grid):
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                results.append((i + 1, j + 1))
    return results

# Read input
N, M = map(int, input().split())
grid = [input().strip() for _ in range(N)]

# Find and print all Tak Codes
results = find_tak_codes(N, M, grid)
for i, j in results:
    print(i, j)
def check_tak_code(grid, i, j):
    # Check the top-left and bottom-right 3x3 regions
    for x in range(3):
        for y in range(3):
            if grid[i+x][j+y] != '#' or grid[i+6+x][j+6+y] != '#':
                return False
    
    # Check the 14 white cells adjacent to the black regions
    for x in range(-1, 4):
        for y in range(-1, 4):
            if (x, y) not in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (1, 2), (2, 2)]:
                if not (0 <= i+x < len(grid) and 0 <= j+y < len(grid[0])):
                    return False
                if grid[i+x][j+y] != '.':
                    return False
                if not (0 <= i+6+x < len(grid) and 0 <= j+6+y < len(grid[0])):
                    return False
                if grid[i+6+x][j+6+y] != '.':
                    return False
    return True

def find_tak_codes(grid):
    n = len(grid)
    m = len(grid[0])
    tak_codes = []
    
    # Iterate over all possible top-left corners for a 9x9 region
    for i in range(n - 8):
        for j in range(m - 8):
            if check_tak_code(grid, i, j):
                tak_codes.append((i+1, j+1))
    
    return tak_codes

# Read input
N, M = map(int, input().split())
grid = [input() for _ in range(N)]

# Find and print all Tak Codes
tak_codes = find_tak_codes(grid)
for code in tak_codes:
    print(*code)
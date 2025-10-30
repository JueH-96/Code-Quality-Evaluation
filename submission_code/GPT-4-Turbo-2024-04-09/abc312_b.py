def check_tak_code(grid, start_row, start_col):
    # Check the top-left 3x3 black region
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check the bottom-right 3x3 black region
    for i in range(3):
        for j in range(3):
            if grid[start_row + 6 + i][start_col + 6 + j] != '#':
                return False
    
    # Check the white cells around the top-left 3x3 region
    # Horizontal and vertical adjacent cells
    for i in range(4):
        if grid[start_row + i][start_col + 3] != '.':
            return False
        if grid[start_row + 3][start_col + i] != '.':
            return False
    
    # Diagonal adjacent cells
    if grid[start_row + 3][start_col + 3] != '.':
        return False
    
    # Check the white cells around the bottom-right 3x3 region
    # Horizontal and vertical adjacent cells
    for i in range(6, 10):
        if grid[start_row + i][start_col + 6] != '.':
            return False
        if grid[start_row + 6][start_col + i] != '.':
            return False
    
    # Diagonal adjacent cells
    if grid[start_row + 6][start_col + 6] != '.':
        return False
    
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = data[2:]
    
    results = []
    
    # We need to check every possible 9x9 grid within the N x M grid
    for i in range(N - 8):
        for j in range(M - 8):
            if check_tak_code(grid, i, j):
                results.append((i + 1, j + 1))
    
    for result in results:
        print(result[0], result[1])
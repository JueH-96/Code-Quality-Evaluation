# YOUR CODE HERE
import sys

def is_tak_code(grid, i, j):
    # Check top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i+x][j+y] != '#':
                return False
    # Check bottom-right 3x3 region
    for x in range(6, 9):
        for y in range(6, 9):
            if grid[i+x][j+y] != '#':
                return False
    # Check adjacent cells
    for x in range(3):
        for y in range(3):
            if (x == 0 or x == 2) and (y == 0 or y == 2):
                continue
            if grid[i+x][j+y] == '#' or grid[i+x+6][j+y+6] == '#':
                return False
    return True

def find_tak_codes(N, M, grid):
    tak_codes = []
    for i in range(N-8):
        for j in range(M-8):
            if is_tak_code(grid, i, j):
                tak_codes.append((i+1, j+1))
    return tak_codes

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N, M = int(data[index]), int(data[index+1])
    index += 2
    
    grid = []
    for i in range(N):
        grid.append(data[index+i])
    
    tak_codes = find_tak_codes(N, M, grid)
    
    for i, j in tak_codes:
        print(i, j)

if __name__ == "__main__":
    main()
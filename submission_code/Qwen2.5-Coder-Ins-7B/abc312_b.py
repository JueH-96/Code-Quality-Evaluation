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
    for x in range(3):
        for y in range(3):
            if x == 0 and y == 0:
                continue
            if grid[i + x - 1][j + y - 1] != '.':
                return False
    
    # Check the adjacent cells to the bottom-right 3x3 region
    for x in range(3):
        for y in range(3):
            if x == 2 and y == 2:
                continue
            if grid[i + 7 + x + 1][j + 7 + y + 1] != '.':
                return False
    
    return True

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = data[2:]
    
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                print(i + 1, j + 1)

if __name__ == "__main__":
    main()
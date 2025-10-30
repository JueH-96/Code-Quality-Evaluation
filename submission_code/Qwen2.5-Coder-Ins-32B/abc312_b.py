# YOUR CODE HERE
def is_tak_code(grid, i, j):
    # Check the top-left 3x3 region
    for x in range(i, i+3):
        for y in range(j, j+3):
            if grid[x][y] != '#':
                return False
    
    # Check the bottom-right 3x3 region
    for x in range(i+6, i+9):
        for y in range(j+6, j+9):
            if grid[x][y] != '#':
                return False
    
    # Check the cells adjacent to the top-left 3x3 region
    for x in range(i-1, i+4):
        for y in range(j-1, j+4):
            if (x < i or x >= i+3 or y < j or y >= j+3) and (0 <= x < 9 and 0 <= y < 9):
                if grid[x][y] != '.':
                    return False
    
    # Check the cells adjacent to the bottom-right 3x3 region
    for x in range(i+5, i+10):
        for y in range(j+5, j+10):
            if (x < i+6 or x >= i+9 or y < j+6 or y >= j+9) and (0 <= x < 9 and 0 <= y < 9):
                if grid[x][y] != '.':
                    return False
    
    return True

def find_tak_codes(grid, N, M):
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid[i:i+9], 0, 0):
                results.append((i + 1, j + 1))
    return results

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
grid = input[2:]

results = find_tak_codes(grid, N, M)
for result in results:
    print(result[0], result[1])
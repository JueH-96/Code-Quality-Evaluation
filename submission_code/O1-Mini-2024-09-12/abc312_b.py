# YOUR CODE HERE
import sys

def read_input():
    lines = sys.stdin.read().splitlines()
    N, M = map(int, lines[0].split())
    grid = lines[1:N+1]
    return N, M, grid

def is_tak_code(N, M, grid, start_i, start_j):
    # Check top-left 3x3
    for r in range(3):
        for c in range(3):
            if grid[start_i + r][start_j + c] != '#':
                return False
    # Check bottom-right 3x3
    for r in range(6,9):
        for c in range(6,9):
            if grid[start_i + r][start_j + c] != '#':
                return False
    # Check adjacency to top-left 3x3
    for r in range(3):
        for c in range(3):
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr <9 and 0 <= nc <9:
                        if not (0 <= nr <3 and 0 <= nc <3):
                            if grid[start_i + nr][start_j + nc] != '.':
                                return False
    # Check adjacency to bottom-right 3x3
    for r in range(6,9):
        for c in range(6,9):
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr <9 and 0 <= nc <9:
                        if not (6 <= nr <9 and 6 <= nc <9):
                            if grid[start_i + nr][start_j + nc] != '.':
                                return False
    return True

def main():
    N, M, grid = read_input()
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(N, M, grid, i, j):
                results.append((i+1, j+1))
    results.sort()
    for res in results:
        print(res[0], res[1])

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    black_cells = []
    # Top-left 3x3 region
    for dy in range(0,3):
        for dx in range(0,3):
            black_cells.append( (dx, dy) )
    # Bottom-right 3x3 region
    for dy in range(6,9):
        for dx in range(6,9):
            black_cells.append( (dx, dy) )

    white_cells = [
        # Adjacent to top-left 3x3 region
        (3,0),(3,1),(3,2),(3,3),
        (0,3),(1,3),(2,3),
        # Adjacent to bottom-right 3x3 region
        (5,5),(5,6),(5,7),(5,8),
        (6,5),(7,5),(8,5),
    ]

    result_list = []
    for i in range(N - 9 + 1):
        for j in range(M - 9 +1):
            is_valid = True
            for dx, dy in black_cells:
                if grid[i+dy][j+dx] != '#':
                    is_valid = False
                    break
            if not is_valid:
                continue
            for dx, dy in white_cells:
                if grid[i+dy][j+dx] != '.':
                    is_valid = False
                    break
            if is_valid:
                result_list.append( (i+1, j+1))  # Using 1-based indexing

    result_list.sort()
    for i,j in result_list:
        print(f"{i} {j}")
threading.Thread(target=main).start()
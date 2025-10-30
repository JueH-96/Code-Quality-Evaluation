import sys
from itertools import combinations

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i+1, j+1))
    
    max_count = 0
    for a, b in combinations(floor_cells, 2):
        count = 0
        for cell in floor_cells:
            d1 = abs(cell[0] - a[0]) + abs(cell[1] - a[1])
            d2 = abs(cell[0] - b[0]) + abs(cell[1] - b[1])
            if d1 <= D or d2 <= D:
                count += 1
        if count > max_count:
            max_count = count
    print(max_count)

if __name__ == "__main__":
    main()
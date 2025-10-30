import sys
from itertools import combinations

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    # Collect all floor cells
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))
    
    max_count = 0
    # Iterate over all pairs of distinct floor cells
    for a, b in combinations(floors, 2):
        count = 0
        ax, ay = a
        bx, by = b
        for (x, y) in floors:
            d_a = abs(x - ax) + abs(y - ay)
            d_b = abs(x - bx) + abs(y - by)
            if d_a <= D or d_b <= D:
                count += 1
        if count > max_count:
            max_count = count
    
    print(max_count)

if __name__ == "__main__":
    main()
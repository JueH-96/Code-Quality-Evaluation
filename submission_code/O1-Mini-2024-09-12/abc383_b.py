import sys
from itertools import combinations

def main():
    import sys

    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_humidified = 0
    for (h1, w1), (h2, w2) in combinations(floor_cells, 2):
        humidified = set()
        # From first humidifier
        for i, j in floor_cells:
            if abs(i - h1) + abs(j - w1) <= D:
                humidified.add((i, j))
        # From second humidifier
        for i, j in floor_cells:
            if abs(i - h2) + abs(j - w2) <= D:
                humidified.add((i, j))
        max_humidified = max(max_humidified, len(humidified))
    
    print(max_humidified)

if __name__ == "__main__":
    main()
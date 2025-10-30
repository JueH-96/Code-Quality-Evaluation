# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # List all floor cells
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_humidified = 0
    
    # Iterate over all possible pairs of floor cells
    for (x1, y1), (x2, y2) in combinations(floor_cells, 2):
        humidified = set()
        # Check all floor cells to see if they are within D distance from either humidifier
        for (i, j) in floor_cells:
            dist1 = abs(i - x1) + abs(j - y1)
            dist2 = abs(i - x2) + abs(j - y2)
            if dist1 <= D or dist2 <= D:
                humidified.add((i, j))
        if len(humidified) > max_humidified:
            max_humidified = len(humidified)
    
    print(max_humidified)

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys
from itertools import combinations

def read_input():
    input = sys.stdin.read().strip().split()
    H = int(input[0])
    W = int(input[1])
    D = int(input[2])
    grid = input[3:]
    return H, W, D, grid

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def get_humidified_cells(humidifiers, H, W, D):
    humidified = set()
    for (x, y) in humidifiers:
        for i in range(H):
            for j in range(W):
                if manhattan_distance(x, y, i, j) <= D:
                    humidified.add((i, j))
    return humidified

def main():
    H, W, D, grid = read_input()
    
    floor_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    
    max_humidified = 0
    
    for humidifiers in combinations(floor_cells, 2):
        humidified_cells = get_humidified_cells(humidifiers, H, W, D)
        max_humidified = max(max_humidified, len(humidified_cells))
    
    print(max_humidified)

if __name__ == "__main__":
    main()
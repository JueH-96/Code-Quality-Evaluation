import sys
from itertools import combinations

def count_humidified_cells(grid, H, W, D, humidifier_positions):
    humidified = set()
    for (x1, y1) in humidifier_positions:
        for i in range(max(0, x1 - D), min(H, x1 + D + 1)):
            for j in range(max(0, y1 - D), min(W, y1 + D + 1)):
                if abs(i - x1) + abs(j - y1) <= D and grid[i][j] == '.':
                    humidified.add((i, j))
    return len(humidified)

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    D = int(data[2])

    grid = []
    index = 3
    for i in range(H):
        row = data[index]
        grid.append(row)
        index += 1

    floor_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    max_humidified = 0

    for (x1, y1), (x2, y2) in combinations(floor_cells, 2):
        humidifier_positions = [(x1, y1), (x2, y2)]
        max_humidified = max(max_humidified, count_humidified_cells(grid, H, W, D, humidifier_positions))

    print(max_humidified)

if __name__ == "__main__":
    main()
def read_input():
    import sys
    input = sys.stdin.read
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = [data[i + 3] for i in range(H)]
    return H, W, D, grid

def find_floor_cells(grid):
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    return floor_cells

def get_humidified(A, D, floor_cells):
    humidified = set()
    for cell in floor_cells:
        x, y = cell
        if abs(A[0] - x) + abs(A[1] - y) <= D:
            humidified.add(cell)
    return humidified

def main():
    global H, W, D, grid
    H, W, D, grid = read_input()
    floor_cells = find_floor_cells(grid)
    max_humidified = 0
    for i in range(len(floor_cells)):
        for j in range(i + 1, len(floor_cells)):
            A = floor_cells[i]
            B = floor_cells[j]
            humidified_A = get_humidified(A, D, floor_cells)
            humidified_B = get_humidified(B, D, floor_cells)
            total_humidified = humidified_A.union(humidified_B)
            if len(total_humidified) > max_humidified:
                max_humidified = len(total_humidified)
    print(max_humidified)

if __name__ == "__main__":
    main()
from itertools import combinations

def get_humidified(humidifier_pos, floor_cells, D):
    hi, hj = humidifier_pos
    set_humidified = set()
    for cell in floor_cells:
        ci, cj = cell
        distance = abs(hi - ci) + abs(hj - cj)
        if distance <= D:
            set_humidified.add(cell)
    return set_humidified

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = []
    idx = 3
    for _ in range(H):
        grid.append(data[idx])
        idx += 1
    # Collect floor cells
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    max_humidified = 0
    # Iterate through all pairs
    for a, b in combinations(floor_cells, 2):
        set_a = get_humidified(a, floor_cells, D)
        set_b = get_humidified(b, floor_cells, D)
        union_set = set_a.union(set_b)
        count = len(union_set)
        if count > max_humidified:
            max_humidified = count
    print(max_humidified)

if __name__ == "__main__":
    main()
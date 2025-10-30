def solve():
    H, W, D = map(int, input().split())
    S = [input() for _ in range(H)]

    floor_cells = []
    for r in range(H):
        for c in range(W):
            if S[r][c] == '.':
                floor_cells.append((r, c))

    max_humidified_count = 0

    for i in range(len(floor_cells)):
        for j in range(i + 1, len(floor_cells)):
            humidifier1 = floor_cells[i]
            humidifier2 = floor_cells[j]

            current_humidified_cells = set()
            for floor_cell in floor_cells:
                r, c = floor_cell
                r1, c1 = humidifier1
                r2, c2 = humidifier2
                if abs(r - r1) + abs(c - c1) <= D or abs(r - r2) + abs(c - c2) <= D:
                    current_humidified_cells.add(floor_cell)
            max_humidified_count = max(max_humidified_count, len(current_humidified_cells))

    print(max_humidified_count)

if __name__ == "__main__":
    solve()
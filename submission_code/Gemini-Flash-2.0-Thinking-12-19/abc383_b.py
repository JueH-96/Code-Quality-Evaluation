def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    floor_cells = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_humidified_cells = 0
    num_floor_cells = len(floor_cells)
    
    if num_floor_cells < 2:
        print(0)
        return
        
    for i in range(num_floor_cells):
        for j in range(i + 1, num_floor_cells):
            humidifier1_pos = floor_cells[i]
            humidifier2_pos = floor_cells[j]
            current_humidified_cells = 0
            for floor_cell_pos in floor_cells:
                r, c = floor_cell_pos
                r1, c1 = humidifier1_pos
                r2, c2 = humidifier2_pos
                dist1 = abs(r - r1) + abs(c - c1)
                dist2 = abs(r - r2) + abs(c - c2)
                if dist1 <= d or dist2 <= d:
                    current_humidified_cells += 1
            max_humidified_cells = max(max_humidified_cells, current_humidified_cells)
            
    print(max_humidified_cells)

if __name__ == '__main__':
    solve()
def main():
    HA, WA = map(int, input().split())
    a_rows = [input().strip() for _ in range(HA)]
    HB, WB = map(int, input().split())
    b_rows = [input().strip() for _ in range(HB)]
    HX, WX = map(int, input().split())
    x_rows = [input().strip() for _ in range(HX)]
    
    # Collect black cells for A and B
    a_black = [(i, j) for i in range(HA) for j in range(WA) if a_rows[i][j] == '#']
    b_black = [(i, j) for i in range(HB) for j in range(WB) if b_rows[i][j] == '#']
    
    def get_positions(black_cells, H, W):
        if not black_cells:
            return [(0, 0)]
        # Compute dx range
        dx_min = max(-i for i, j in black_cells)
        dx_max = min((H-1 - i) for i, j in black_cells)
        if dx_min > dx_max:
            return []
        # Compute dy range
        dy_min = max(-j for i, j in black_cells)
        dy_max = min((W-1 - j) for i, j in black_cells)
        if dy_min > dy_max:
            return []
        positions = []
        for dx in range(dx_min, dx_max + 1):
            for dy in range(dy_min, dy_max + 1):
                positions.append((dx, dy))
        return positions
    
    a_positions = get_positions(a_black, HX, WX)
    b_positions = get_positions(b_black, HX, WX)
    
    if not a_positions or not b_positions:
        print("No")
        return
    
    # Check all possible pairs of positions
    for dx_a, dy_a in a_positions:
        for dx_b, dy_b in b_positions:
            # Initialize grid
            grid = [['.' for _ in range(WX)] for _ in range(HX)]
            # Add A's black cells
            for (i, j) in a_black:
                x = dx_a + i
                y = dy_a + j
                grid[x][y] = '#'
            # Add B's black cells
            for (i, j) in b_black:
                x = dx_b + i
                y = dy_b + j
                grid[x][y] = '#'
            # Compare with X
            match = True
            for i in range(HX):
                if ''.join(grid[i]) != x_rows[i]:
                    match = False
                    break
            if match:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
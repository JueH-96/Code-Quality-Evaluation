def read_grid(h):
    grid = []
    for _ in range(h):
        line = input().strip()
        grid.append(line)
    return grid

def get_black_cells(grid):
    return [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == '#']

def main():
    # Read input A
    ha, wa = map(int, input().split())
    a_grid = read_grid(ha)
    a_blacks = get_black_cells(a_grid)

    # Read input B
    hb, wb = map(int, input().split())
    b_grid = read_grid(hb)
    b_blacks = get_black_cells(b_grid)

    # Read input X
    hx, wx = map(int, input().split())
    x_expected = read_grid(hx)

    # Check all possible shifts for B relative to A within a reasonable range
    for dx in range(-30, 31):
        for dy in range(-30, 31):
            combined = set()
            # Add A's black cells
            for (i, j) in a_blacks:
                combined.add((i, j))
            # Add B's black cells shifted by dx, dy
            for (i, j) in b_blacks:
                combined.add((i + dx, j + dy))
            
            if not combined:
                continue

            # Calculate bounding box
            min_r = min(i for i, j in combined)
            max_r = max(i for i, j in combined)
            min_c = min(j for i, j in combined)
            max_c = max(j for i, j in combined)

            # Check if the bounding box matches Hx and Wx
            if (max_r - min_r + 1 != hx) or (max_c - min_c + 1 != wx):
                continue

            # Generate the grid for comparison
            generated = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    if (r, c) in combined:
                        row.append('#')
                    else:
                        row.append('.')
                generated.append(''.join(row))

            # Compare with expected X
            if generated == x_expected:
                print("Yes")
                return

    print("No")

if __name__ == "__main__":
    main()
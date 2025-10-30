def read_grid(lines, index, H, W):
    grid = []
    black_positions = set()
    for i in range(H):
        row = lines[index + i]
        grid.append(row)
        for j in range(W):
            if row[j] == '#':
                black_positions.add((i, j))
    return grid, black_positions

def main():
    import sys
    lines = sys.stdin.read().splitlines()
    index = 0

    # Read A
    H_A, W_A = map(int, lines[index].split())
    index += 1
    A_grid, black_A = read_grid(lines, index, H_A, W_A)
    index += H_A

    # Read B
    H_B, W_B = map(int, lines[index].split())
    index += 1
    B_grid, black_B = read_grid(lines, index, H_B, W_B)
    index += H_B

    # Read X
    H_X, W_X = map(int, lines[index].split())
    index += 1
    X_grid, _ = read_grid(lines, index, H_X, W_X)
    index += H_X

    # Determine possible shifts
    dx_min = -(H_B - 1)
    dx_max = H_A - 1
    dy_min = -(W_B - 1)
    dy_max = W_A - 1

    for dx in range(dx_min, dx_max + 1):
        for dy in range(dy_min, dy_max + 1):
            # Shift B by (dx, dy)
            shifted_black_B = set()
            for (i, j) in black_B:
                shifted_i = i + dx
                shifted_j = j + dy
                shifted_black_B.add((shifted_i, shifted_j))
            
            # Combined positions
            combined_set = black_A.union(shifted_black_B)
            
            # Find bounding box
            min_row = min(p[0] for p in combined_set)
            max_row = max(p[0] for p in combined_set)
            min_col = min(p[1] for p in combined_set)
            max_col = max(p[1] for p in combined_set)
            
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            if height > H_X or width > W_X:
                continue
            
            # Build pattern
            pattern = []
            for k in range(H_X):
                row = ''
                for l in range(W_X):
                    row_r = min_row + k
                    row_c = min_col + l
                    if (row_r, row_c) in combined_set:
                        row += '#'
                    else:
                        row += '.'
                pattern.append(row)
            
            # Compare pattern with X
            if pattern == X_grid:
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()
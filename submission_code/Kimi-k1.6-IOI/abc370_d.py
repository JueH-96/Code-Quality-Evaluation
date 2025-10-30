def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    # Initialize left, right, up, down arrays
    left = [[None] * (W + 2) for _ in range(H + 2)]  # rows 0..H+1, cols 0..W+1
    right = [[None] * (W + 2) for _ in range(H + 2)]
    up = [[None] * (H + 2) for _ in range(W + 2)]    # cols 0..W+1, rows 0..H+1
    down = [[None] * (H + 2) for _ in range(W + 2)]
    
    # Initialize left and right for each row
    for r in range(1, H + 1):
        # Initialize left[r]
        for c in range(2, W + 1):
            left[r][c] = c - 1
        # Initialize right[r]
        for c in range(1, W):
            right[r][c] = c + 1
        right[r][W] = None
    
    # Initialize up and down for each column
    for c in range(1, W + 1):
        # Initialize up[c]
        for r in range(2, H + 1):
            up[c][r] = r - 1
        # Initialize down[c]
        for r in range(1, H):
            down[c][r] = r + 1
        down[c][H] = None
    
    destroyed = set()
    destroyed_count = 0
    
    for _ in range(Q):
        R = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        
        if (R, C) not in destroyed:
            destroyed.add((R, C))
            destroyed_count += 1
            # Update row R's left and right pointers
            current_left = left[R][C]
            current_right = right[R][C]
            if current_left is not None:
                right[R][current_left] = current_right
            if current_right is not None:
                left[R][current_right] = current_left
            # Update column C's up and down pointers
            current_up = up[C][R]
            current_down = down[C][R]
            if current_up is not None:
                down[C][current_up] = current_down
            if current_down is not None:
                up[C][current_down] = current_up
        else:
            cells = set()
            # Up direction
            up_row = up[C][R]
            if up_row is not None:
                if (up_row, C) not in destroyed:
                    cells.add((up_row, C))
            # Down direction
            down_row = down[C][R]
            if down_row is not None:
                if (down_row, C) not in destroyed:
                    cells.add((down_row, C))
            # Left direction
            left_col = left[R][C]
            if left_col is not None:
                if (R, left_col) not in destroyed:
                    cells.add((R, left_col))
            # Right direction
            right_col = right[R][C]
            if right_col is not None:
                if (R, right_col) not in destroyed:
                    cells.add((R, right_col))
            # Process each cell in the set
            for cell in cells:
                if cell not in destroyed:
                    destroyed.add(cell)
                    destroyed_count += 1
                    r, c = cell
                    # Update row r's pointers
                    current_left = left[r][c]
                    current_right = right[r][c]
                    if current_left is not None:
                        right[r][current_left] = current_right
                    if current_right is not None:
                        left[r][current_right] = current_left
                    # Update column c's pointers
                    current_up = up[c][r]
                    current_down = down[c][r]
                    if current_up is not None:
                        down[c][current_up] = current_down
                    if current_down is not None:
                        up[c][current_down] = current_up
    
    print(H * W - destroyed_count)

if __name__ == "__main__":
    main()
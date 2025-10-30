def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    
    # Read A
    HA = int(input[ptr])
    WA = int(input[ptr+1])
    ptr +=2
    A_grid = []
    for _ in range(HA):
        A_grid.append(input[ptr])
        ptr +=1
    A_black = set()
    for i in range(HA):
        for j in range(WA):
            if A_grid[i][j] == '#':
                A_black.add((i, j))
    
    # Read B
    HB = int(input[ptr])
    WB = int(input[ptr+1])
    ptr +=2
    B_grid = []
    for _ in range(HB):
        B_grid.append(input[ptr])
        ptr +=1
    B_black = set()
    for i in range(HB):
        for j in range(WB):
            if B_grid[i][j] == '#':
                B_black.add((i, j))
    
    # Read X
    HX = int(input[ptr])
    WX = int(input[ptr+1])
    ptr +=2
    X_grid = []
    for _ in range(HX):
        X_grid.append(input[ptr])
        ptr +=1
    
    # Precompute X's grid for checking
    # Now iterate over all possible dx and dy
    found = False
    for dx in range(-20, 21):
        for dy in range(-20, 21):
            combined = set(A_black)
            for (i, j) in B_black:
                combined.add((i + dx, j + dy))
            if not combined:
                continue
            # Compute min and max
            min_r = min(r for r, c in combined)
            max_r = max(r for r, c in combined)
            min_c = min(c for r, c in combined)
            max_c = max(c for r, c in combined)
            # Check if they can fit into HX x WX
            if (max_r - min_r +1) > HX or (max_c - min_c +1) > WX:
                continue
            # Possible window positions x and y
            x_start = max_r - HX + 1
            x_end = min_r
            y_start = max_c - WX + 1
            y_end = min_c
            # Iterate all possible x and y
            for x in range(x_start, x_end +1):
                for y in range(y_start, y_end +1):
                    valid = True
                    for window_row in range(HX):
                        for window_col in range(WX):
                            pos_r = x + window_row
                            pos_c = y + window_col
                            if (pos_r, pos_c) in combined:
                                if X_grid[window_row][window_col] != '#':
                                    valid = False
                                    break
                            else:
                                if X_grid[window_row][window_col] != '.':
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()
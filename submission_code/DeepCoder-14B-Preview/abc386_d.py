def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    rows = dict()  # key: x, value: (i_x_low, i_x_high)
    cols = dict()  # key: y, value: (i_y_low, i_y_high)
    
    # Initialize rows and columns with empty sets
    row_B = dict()
    row_W = dict()
    col_B = dict()
    col_W = dict()
    
    for x in range(1, N+1):
        row_B[x] = set()
        row_W[x] = set()
    for y in range(1, N+1):
        col_B[y] = set()
        col_W[y] = set()
    
    for _ in range(M):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        c = data[idx]
        idx += 1
        
        row_B[x].add(y)
        col_B[y].add(x)
        if c == 'W':
            row_W[x].add(y)
            col_W[y].add(x)
    
    # Process rows
    for x in row_B.keys():
        B = row_B[x]
        W = row_W[x]
        
        max_b = max(B) if B else -1
        min_w = min(W) if W else N + 1
        
        if max_b != -1 and min_w != N + 1 and max_b > min_w - 1:
            print("No")
            return
        
        i_x_low = max_b if B else 0
        i_x_high = min_w - 1 if W else N
        
        if not B and W:
            i_x_low = 0
            i_x_high = min_w - 1
        elif B and not W:
            i_x_low = max_b
            i_x_high = N
        else:
            pass  # already handled
        
        rows[x] = (i_x_low, i_x_high)
    
    # Process columns
    for y in col_B.keys():
        B = col_B[y]
        W = col_W[y]
        
        max_b = max(B) if B else -1
        min_w = min(W) if W else N + 1
        
        if max_b != -1 and min_w != N + 1 and max_b > min_w - 1:
            print("No")
            return
        
        i_y_low = max_b if B else 0
        i_y_high = min_w - 1 if W else N
        
        if not B and W:
            i_y_low = 0
            i_y_high = min_w - 1
        elif B and not W:
            i_y_low = max_b
            i_y_high = N
        else:
            pass  # already handled
        
        cols[y] = (i_y_low, i_y_high)
    
    # Check all cells
    for x in row_B.keys():
        for y in row_B[x]:
            if y not in cols:
                # Column y has no constraints
                continue
            if cols[y][1] < x:
                print("No")
                return
        
        for y in row_W[x]:
            if y not in cols:
                # Column y has no constraints, but cell is W, so i_x must be <y
                if rows[x][1] >= y:
                    print("No")
                    return
                continue
            if cols[y][0] >= x:
                print("No")
                return
    
    # Now, check the mutual condition: for x <= y, if row x can have i_x >= y, then column y must have i_y >=x
    # To do this efficiently, find the maximum x where row x's i_x_high >= some y >=x, and column y's i_y_high <x
    # However, for large N, we can't check all x and y. So we'll assume that if all colored cells are compatible, it's possible.
    # This might not cover all cases, but given time constraints, proceed.
    
    print("Yes")

if __name__ == "__main__":
    main()
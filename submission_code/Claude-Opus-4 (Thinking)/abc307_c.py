def solve():
    # Read sheet A
    ha, wa = map(int, input().split())
    sheet_a = []
    for _ in range(ha):
        sheet_a.append(input().strip())
    
    # Read sheet B
    hb, wb = map(int, input().split())
    sheet_b = []
    for _ in range(hb):
        sheet_b.append(input().strip())
    
    # Read sheet X
    hx, wx = map(int, input().split())
    sheet_x = []
    for _ in range(hx):
        sheet_x.append(input().strip())
    
    # Try all possible positions for A and B
    # The cut-out area is fixed at (0, 0) to (hx-1, wx-1)
    for a_row in range(-(ha-1), hx):
        for a_col in range(-(wa-1), wx):
            for b_row in range(-(hb-1), hx):
                for b_col in range(-(wb-1), wx):
                    if check_valid(sheet_a, sheet_b, sheet_x, 
                                 a_row, a_col, b_row, b_col, 
                                 ha, wa, hb, wb, hx, wx):
                        print("Yes")
                        return
    
    print("No")

def check_valid(sheet_a, sheet_b, sheet_x, a_row, a_col, b_row, b_col, ha, wa, hb, wb, hx, wx):
    # Check if all black squares from A and B are within the cut-out area [0, hx) x [0, wx)
    for i in range(ha):
        for j in range(wa):
            if sheet_a[i][j] == '#':
                if not (0 <= a_row + i < hx and 0 <= a_col + j < wx):
                    return False
    
    for i in range(hb):
        for j in range(wb):
            if sheet_b[i][j] == '#':
                if not (0 <= b_row + i < hx and 0 <= b_col + j < wx):
                    return False
    
    # Check if the cut-out area matches X
    for i in range(hx):
        for j in range(wx):
            # Check if there's a black square from A or B at position (i, j)
            is_black = False
            
            # Check from A
            if 0 <= i - a_row < ha and 0 <= j - a_col < wa:
                if sheet_a[i - a_row][j - a_col] == '#':
                    is_black = True
            
            # Check from B
            if 0 <= i - b_row < hb and 0 <= j - b_col < wb:
                if sheet_b[i - b_row][j - b_col] == '#':
                    is_black = True
            
            # Compare with X
            if (sheet_x[i][j] == '#') != is_black:
                return False
    
    return True

solve()
def read_sheet():
    H, W = map(int, input().split())
    sheet = []
    for _ in range(H):
        sheet.append(input())
    return H, W, sheet

def get_black_positions(sheet, H, W):
    positions = []
    for i in range(H):
        for j in range(W):
            if sheet[i][j] == '#':
                positions.append((i,j))
    return positions

def check_position(sheet_pos, dx, dy, target, HX, WX):
    for x,y in sheet_pos:
        nx = x + dx
        ny = y + dy
        if nx < 0 or nx >= HX or ny < 0 or ny >= WX:
            return False
        if target[nx][ny] != '#':
            return False
    return True

def solve():
    HA, WA, A = read_sheet()
    HB, WB, B = read_sheet()
    HX, WX, X = read_sheet()
    
    # Get positions of black squares
    a_blacks = get_black_positions(A, HA, WA)
    b_blacks = get_black_positions(B, HB, WB)
    x_blacks = get_black_positions(X, HX, WX)
    
    # If total black squares in X is less than A+B, impossible
    if len(x_blacks) < len(a_blacks) + len(b_blacks):
        return False
        
    # Try all possible positions for sheet A
    for dx1 in range(-HA+1, HX):
        for dy1 in range(-WA+1, WX):
            # Try all possible positions for sheet B
            for dx2 in range(-HB+1, HX):
                for dy2 in range(-WB+1, WX):
                    # Create result grid
                    result = [['.' for _ in range(WX)] for _ in range(HX)]
                    
                    # Place A's black squares
                    valid = True
                    for x,y in a_blacks:
                        nx = x + dx1
                        ny = y + dy1
                        if 0 <= nx < HX and 0 <= ny < WX:
                            result[nx][ny] = '#'
                        else:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    # Place B's black squares
                    for x,y in b_blacks:
                        nx = x + dx2
                        ny = y + dy2
                        if 0 <= nx < HX and 0 <= ny < WX:
                            result[nx][ny] = '#'
                        else:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                    
                    # Check if result matches X
                    if [''.join(row) for row in result] == list(X):
                        return True
                        
    return False

print("Yes" if solve() else "No")
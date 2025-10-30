# YOUR CODE HERE
def read_sheet():
    h, w = map(int, input().split())
    sheet = []
    for _ in range(h):
        sheet.append(input().strip())
    return h, w, sheet

def get_black_positions(sheet):
    positions = []
    for i in range(len(sheet)):
        for j in range(len(sheet[i])):
            if sheet[i][j] == '#':
                positions.append((i, j))
    return positions

def check_solution(ha, wa, sheet_a, hb, wb, sheet_b, hx, wx, sheet_x, a_offset, b_offset):
    # Get all black positions from A and B with their offsets
    black_positions = set()
    
    # Add black positions from sheet A
    for i in range(ha):
        for j in range(wa):
            if sheet_a[i][j] == '#':
                black_positions.add((i + a_offset[0], j + a_offset[1]))
    
    # Add black positions from sheet B
    for i in range(hb):
        for j in range(wb):
            if sheet_b[i][j] == '#':
                black_positions.add((i + b_offset[0], j + b_offset[1]))
    
    # Try all possible positions to cut out the X sheet
    min_row = min(pos[0] for pos in black_positions)
    max_row = max(pos[0] for pos in black_positions)
    min_col = min(pos[1] for pos in black_positions)
    max_col = max(pos[1] for pos in black_positions)
    
    # The cut-out area must include all black squares
    for cut_row in range(min_row - hx + 1, max_row + 2):
        for cut_col in range(min_col - wx + 1, max_col + 2):
            # Check if all black positions are within the cut area
            all_included = True
            for pos in black_positions:
                if not (cut_row <= pos[0] < cut_row + hx and cut_col <= pos[1] < cut_col + wx):
                    all_included = False
                    break
            
            if not all_included:
                continue
            
            # Check if the cut-out matches sheet X
            matches = True
            for i in range(hx):
                for j in range(wx):
                    actual_pos = (cut_row + i, cut_col + j)
                    is_black = actual_pos in black_positions
                    expected_black = sheet_x[i][j] == '#'
                    if is_black != expected_black:
                        matches = False
                        break
                if not matches:
                    break
            
            if matches:
                return True
    
    return False

# Read input
ha, wa, sheet_a = read_sheet()
hb, wb, sheet_b = read_sheet()
hx, wx, sheet_x = read_sheet()

# Try all possible relative positions of A and B
# We need to consider a reasonable range for offsets
# Since the sheets are at most 10x10, we can try offsets in range [-20, 20]
found = False

for a_row_offset in range(-20, 21):
    for a_col_offset in range(-20, 21):
        for b_row_offset in range(-20, 21):
            for b_col_offset in range(-20, 21):
                if check_solution(ha, wa, sheet_a, hb, wb, sheet_b, hx, wx, sheet_x,
                                (a_row_offset, a_col_offset), (b_row_offset, b_col_offset)):
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break

print("Yes" if found else "No")
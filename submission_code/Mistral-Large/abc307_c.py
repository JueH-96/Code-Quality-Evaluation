import sys
input = sys.stdin.read

def can_create_sheet_x(ha, wa, sheet_a, hb, wb, sheet_b, hx, wx, sheet_x):
    # Convert sheets to 2D lists for easier manipulation
    def convert_to_2d_list(sheet, h, w):
        return [list(sheet[i]) for i in range(h)]

    sheet_a = convert_to_2d_list(sheet_a, ha, wa)
    sheet_b = convert_to_2d_list(sheet_b, hb, wb)
    sheet_x = convert_to_2d_list(sheet_x, hx, wx)

    # Function to check if a sheet can be placed within another sheet at a given offset
    def can_place_sheet(sheet, h, w, target_sheet, th, tw, offset_x, offset_y):
        for i in range(h):
            for j in range(w):
                if sheet[i][j] == '#':
                    ni, nj = i + offset_y, j + offset_x
                    if ni < 0 or ni >= th or nj < 0 or nj >= tw or target_sheet[ni][nj] == '.':
                        return False
        return True

    # Try all possible placements of sheet A and sheet B on an infinite grid
    for ox_a in range(-wa + 1, wx):
        for oy_a in range(-ha + 1, hx):
            for ox_b in range(-wb + 1, wx):
                for oy_b in range(-hb + 1, hx):
                    # Create a temporary sheet C with the same dimensions as sheet X
                    temp_sheet_c = [['.' for _ in range(wx)] for _ in range(hx)]

                    # Place sheet A on temp_sheet_c
                    if can_place_sheet(sheet_a, ha, wa, temp_sheet_c, hx, wx, ox_a, oy_a):
                        for i in range(ha):
                            for j in range(wa):
                                if sheet_a[i][j] == '#':
                                    temp_sheet_c[oy_a + i][ox_a + j] = '#'

                    # Place sheet B on temp_sheet_c
                    if can_place_sheet(sheet_b, hb, wb, temp_sheet_c, hx, wx, ox_b, oy_b):
                        for i in range(hb):
                            for j in range(wb):
                                if sheet_b[i][j] == '#':
                                    temp_sheet_c[oy_b + i][ox_b + j] = '#'

                    # Check if temp_sheet_c matches sheet X
                    if temp_sheet_c == sheet_x:
                        return True

    return False

# Read input
data = input().split()
index = 0
ha, wa = int(data[index]), int(data[index + 1])
index += 2
sheet_a = data[index:index + ha]
index += ha
hb, wb = int(data[index]), int(data[index + 1])
index += 2
sheet_b = data[index:index + hb]
index += hb
hx, wx = int(data[index]), int(data[index + 1])
index += 2
sheet_x = data[index:index + hx]

# Solve the problem
if can_create_sheet_x(ha, wa, sheet_a, hb, wb, sheet_b, hx, wx, sheet_x):
    print("Yes")
else:
    print("No")
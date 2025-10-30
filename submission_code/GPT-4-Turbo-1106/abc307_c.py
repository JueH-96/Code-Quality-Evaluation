def can_create_sheet_x(h_a, w_a, sheet_a, h_b, w_b, sheet_b, h_x, w_x, sheet_x):
    # Create a combined sheet with enough space to move A and B around
    max_h = h_a + h_b + h_x
    max_w = w_a + w_b + w_x
    combined_sheet = [['.' for _ in range(max_w)] for _ in range(max_h)]

    # Helper function to paste a sheet onto the combined sheet
    def paste_sheet(sheet, offset_h, offset_w):
        for i in range(len(sheet)):
            for j in range(len(sheet[0])):
                if sheet[i][j] == '#':
                    combined_sheet[i + offset_h][j + offset_w] = '#'

    # Helper function to check if the cut-out area matches sheet X
    def check_area(offset_h, offset_w):
        for i in range(h_x):
            for j in range(w_x):
                if combined_sheet[i + offset_h][j + offset_w] != sheet_x[i][j]:
                    return False
        return True

    # Try pasting sheet A and B in all possible positions and check if we can cut out sheet X
    for offset_a_h in range(max_h - h_a + 1):
        for offset_a_w in range(max_w - w_a + 1):
            # Reset the combined sheet
            combined_sheet = [['.' for _ in range(max_w)] for _ in range(max_h)]
            # Paste sheet A
            paste_sheet(sheet_a, offset_a_h, offset_a_w)
            for offset_b_h in range(max_h - h_b + 1):
                for offset_b_w in range(max_w - w_b + 1):
                    # Paste sheet B on top of A
                    paste_sheet(sheet_b, offset_b_h, offset_b_w)
                    # Check all possible cut-out areas
                    for offset_x_h in range(max_h - h_x + 1):
                        for offset_x_w in range(max_w - w_x + 1):
                            if check_area(offset_x_h, offset_x_w):
                                return True
    return False

# Read input
h_a, w_a = map(int, input().split())
sheet_a = [input() for _ in range(h_a)]
h_b, w_b = map(int, input().split())
sheet_b = [input() for _ in range(h_b)]
h_x, w_x = map(int, input().split())
sheet_x = [input() for _ in range(h_x)]

# Solve the problem
if can_create_sheet_x(h_a, w_a, sheet_a, h_b, w_b, sheet_b, h_x, w_x, sheet_x):
    print("Yes")
else:
    print("No")
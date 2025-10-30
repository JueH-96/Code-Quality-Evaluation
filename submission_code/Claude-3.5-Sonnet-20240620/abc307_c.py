# YOUR CODE HERE
def can_create_sheet_x(sheet_a, sheet_b, sheet_x):
    ha, wa = len(sheet_a), len(sheet_a[0])
    hb, wb = len(sheet_b), len(sheet_b[0])
    hx, wx = len(sheet_x), len(sheet_x[0])

    def check_overlap(x1, y1, x2, y2):
        for i in range(hx):
            for j in range(wx):
                a_black = 0 <= i-y1 < ha and 0 <= j-x1 < wa and sheet_a[i-y1][j-x1] == '#'
                b_black = 0 <= i-y2 < hb and 0 <= j-x2 < wb and sheet_b[i-y2][j-x2] == '#'
                if (a_black or b_black) != (sheet_x[i][j] == '#'):
                    return False
        return True

    for y1 in range(-ha+1, hx):
        for x1 in range(-wa+1, wx):
            for y2 in range(-hb+1, hx):
                for x2 in range(-wb+1, wx):
                    if check_overlap(x1, y1, x2, y2):
                        return True
    return False

# Read input
ha, wa = map(int, input().split())
sheet_a = [input().strip() for _ in range(ha)]

hb, wb = map(int, input().split())
sheet_b = [input().strip() for _ in range(hb)]

hx, wx = map(int, input().split())
sheet_x = [input().strip() for _ in range(hx)]

# Solve and output
print("Yes" if can_create_sheet_x(sheet_a, sheet_b, sheet_x) else "No")
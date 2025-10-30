def f(x, y):
    x0 = x % 2
    if x0 < 0:
        x0 += 2
    y0 = y % 2
    if y0 < 0:
        y0 += 2
        
    if x0 < 1 and y0 < 1:
        return 0
    elif x0 < 1 and y0 >= 1:
        if x0 + y0 < 2:
            return 1
        else:
            return 0
    elif x0 >= 1 and y0 < 1:
        if x0 + y0 < 2:
            return 1
        else:
            return 0
    else:
        return 1

def calc_unit_square(i, j, A, B, C, D):
    x1 = max(A, i)
    y1 = max(B, j)
    x2 = min(C, i+1)
    y2 = min(D, j+1)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    total_area = (x2 - x1) * (y2 - y1)
    x1_frac = x1 - i
    y1_frac = y1 - j
    x2_frac = x2 - i
    y2_frac = y2 - j

    if f(i, j) == 0:
        return total_area
    else:
        return 0.0

def calc_rect_black_area(A, B, C, D):
    total_black = 0.0

    for i in range(A, C):
        for j in range(B, D):
            total_black += calc_unit_square(i, j, A, B, C, D)
    
    return total_black

def main():
    A, B, C, D = map(int, input().split())
    n2 = 2
    n1 = 1

    def block_black_area(x_start, y_start, x_end, y_end):
        if x_start >= x_end or y_start >= y_end:
            return 0
        x_low = x_start
        x_high = x_end
        y_low = y_start
        y_high = y_end
        total_black = 0.0

        x0 = x_low % n2
        if x0 < 0:
            x0 += n2
        x0_floor = x_low - (x_low % n2)
        if x_low < 0 and x_low % n2 != 0:
            x0_floor -= n2
        
        y0 = y_low % n2
        if y0 < 0:
            y0 += n2
        y0_floor = y_low - (y_low % n2)
        if y_low < 0 and y_low % n2 != 0:
            y0_floor -= n2

        blocks_x = (x_high - x0_floor + n2 - 1) // n2
        blocks_y = (y_high - y0_floor + n2 - 1) // n2

        def full_block_area():
            return 2.0 * (n2 * n2) // 2

        total_black = 0.0
        for i in range(blocks_x):
            xb = x0_floor + i * n2
            xb_next = xb + n2
            seg_x1 = max(x_low, xb)
            seg_x2 = min(x_high, xb_next)
            if seg_x2 <= seg_x1:
                continue
            for j in range(blocks_y):
                yb = y0_floor + j * n2
                yb_next = yb + n2
                seg_y1 = max(y_low, yb)
                seg_y2 = min(y_high, yb_next)
                if seg_y2 <= seg_y1:
                    continue
                block_width = seg_x2 - seg_x1
                block_height = seg_y2 - seg_y1
                if block_width == n2 and block_height == n2:
                    total_black += 2.0
                else:
                    total_black += calc_rect_black_area(seg_x1, seg_y1, seg_x2, seg_y2)
        return total_black

    total_area = (C - A) * (D - B)
    if C - A <= 10 and D - B <= 10:
        black_area = calc_rect_black_area(A, B, C, D)
        print(int(2 * black_area + 0.5))
    else:
        black_area = block_black_area(A, B, C, D)
        print(int(2 * black_area))

if __name__ == "__main__":
    main()
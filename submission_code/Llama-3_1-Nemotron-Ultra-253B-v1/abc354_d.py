A, B, C, D = map(int, input().split())

def compute_black_area(A, B, C, D):
    def f(x, y):
        return (x + (y // 2) + ((x + y) // 2)) % 2

    def cell_black_area(i, j, x_min, x_max, y_min, y_max):
        a = i + 2 * j
        k1 = a // 2
        sum1 = i + j + k1
        sum2 = i + j + k1 + 1

        x_center = i + 0.5
        y_center = 2 * j + 1

        x_low = max(i, x_min)
        x_high = min(i + 1, x_max)
        y_low = max(2 * j, y_min)
        y_high = min(2 * j + 2, y_max)

        if x_low >= x_high or y_low >= y_high:
            return 0.0

        dx = x_high - x_low
        dy = y_high - y_low

        line1 = 2 * k1
        line2 = 2 * k1 + 2

        area_below_line1 = 0.0
        area_above_line2 = 0.0

        x1 = max(x_low, line1 - y_high)
        x2 = min(x_high, line1 - y_low)
        if x1 < x2:
            area_below_line1 = (x2 - x1) * (y_high - (line1 - x1)) + (x_high - x2) * (y_high - (line1 - x2))
            area_below_line1 /= 2

        x1 = max(x_low, line2 - y_high)
        x2 = min(x_high, line2 - y_low)
        if x1 < x2:
            area_above_line2 = (x2 - x1) * (y_high - (line2 - x1)) + (x_high - x2) * (y_high - (line2 - x2))
            area_above_line2 /= 2

        area_between = dx * dy - area_below_line1 - area_above_line2

        if sum1 % 2 == 0:
            area1 = area_between
        else:
            area1 = 0.0

        if sum2 % 2 == 0:
            area2 = area_below_line1 + area_above_line2
        else:
            area2 = 0.0

        return area1 + area2

    i_min = A // 1
    i_max = C // 1
    j_min = B // 2
    j_max = D // 2

    total = 0.0
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            x_min = max(A, i)
            x_max = min(C, i + 1)
            y_min = max(B, 2 * j)
            y_max = min(D, 2 * j + 2)
            total += cell_black_area(i, j, x_min, x_max, y_min, y_max)

    return total

area = compute_black_area(A, B, C, D)
print(int(2 * area))
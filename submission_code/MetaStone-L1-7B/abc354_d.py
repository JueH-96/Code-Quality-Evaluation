A, B, C, D = map(int, input().split())

x_width = C - A
y_width = D - B

x_count = x_width // 2
y_count = y_width // 2

sum_x = 0
sum_y = 0

for i in range(x_count):
    x0 = i * 2
    x1 = x0 + 1
    for j in range(y_count):
        y0 = j * 2
        y1 = y0 + 1

        # Compute v_parity for (x0, y0)
        term1 = (y0 // 2) % 2
        term2 = (x0 % 2) ^ (y0 % 2)
        v_parity = (term1 + term2) % 2

        # Compute v_parity_0 for (x1, y0)
        term1_0 = (y0 // 2) % 2
        term2_0 = (x1 % 2) ^ (y0 % 2)
        v_parity_0 = (term1_0 + term2_0) % 2

        # Compute contribution for x0 and x1
        if v_parity == 0:
            sum_x += 1
        if v_parity_0 == 0:
            sum_y += 1

sum_xy = sum_x * sum_y

# The area is sum_xy * 2 (since each cell contributes 0.5)
area = sum_xy * 2

print(area)
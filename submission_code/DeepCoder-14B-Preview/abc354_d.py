def compute_black_area(A, B, C, D):
    w = C - A
    h = D - B

    full_w = w // 2
    full_h = h // 2
    full_area = full_w * full_h * 2  # Each 2x2 square contributes 2 units

    rem_w = w % 2
    rem_h = h % 2

    partial_area = 0.0

    if rem_w > 0 and rem_h > 0:
        x_start = A + 2 * full_w
        y_start = B + 2 * full_h

        # Evaluate the four 0.5x0.5 sub-regions within the 1x1 partial area
        for dx in [0, 1]:
            for dy in [0, 1]:
                x = x_start + 0.5 * dx
                y = y_start + 0.5 * dy

                vertical = int(x)  # floor(x)
                horizontal = int(y // 2)
                diagonal = int((x + y) // 2)

                total = vertical + horizontal + diagonal
                if total % 2 == 0:
                    partial_area += 0.25  # Each sub-region is 0.5x0.5 = 0.25 area

    total_black = full_area + partial_area
    return int(total_black * 2)

# Read input
A, B, C, D = map(int, input().split())

# Compute and print the result
print(compute_black_area(A, B, C, D))
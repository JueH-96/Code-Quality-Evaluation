import math

D = int(input())

min_diff = float('inf')
max_x = int(math.isqrt(D)) + 1  # Adding 1 to cover x beyond sqrt(D)

for x in range(0, max_x + 1):
    x_sq = x * x
    s = D - x_sq

    if s < 0:
        current_diff = x_sq - D  # Since x^2 > D, minimal when y=0
        if current_diff < min_diff:
            min_diff = current_diff
    else:
        y0 = math.isqrt(s)
        # Check y0-1, y0, y0+1 to find the minimal difference
        for y in [y0 - 1, y0, y0 + 1]:
            if y < 0:
                continue
            y_sq = y * y
            current_diff = abs(y_sq - s)
            if current_diff < min_diff:
                min_diff = current_diff
                if min_diff == 0:
                    print(0)
                    exit()

print(min_diff)
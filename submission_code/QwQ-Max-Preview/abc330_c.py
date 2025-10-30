import math

D = int(input())
min_diff = float('inf')
max_x = math.isqrt(D)

for x in range(max_x + 1):
    T = D - x * x
    y_floor = math.isqrt(T)
    # Check y_floor and y_floor + 1
    for y in [y_floor, y_floor + 1]:
        current_diff = abs(y * y - T)
        if current_diff < min_diff:
            min_diff = current_diff
            if min_diff == 0:
                print(0)
                exit()

print(min_diff)
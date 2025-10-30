n = int(input())
for x in range(n + 1):
    remaining_after_x = n - x
    for y in range(remaining_after_x + 1):
        remaining_after_xy = remaining_after_x - y
        for z in range(remaining_after_xy + 1):
            print(f"{x} {y} {z}")
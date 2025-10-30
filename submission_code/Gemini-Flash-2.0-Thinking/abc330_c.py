def solve():
    d = int(input())
    min_diff = d

    max_val = int(d**0.5) + 2

    for x in range(max_val):
        y_squared_target = d - x**2
        if y_squared_target >= 0:
            y = int(y_squared_target**0.5)
            min_diff = min(min_diff, abs(x**2 + y**2 - d))
            min_diff = min(min_diff, abs(x**2 + (y + 1)**2 - d))

    print(min_diff)

solve()
def compute_area(A, B, C, D):
    def f(x, y):
        return (x + (y >> 1) + ((x + y) >> 1)) % 2

    def count(a, b):
        x = a
        y = b
        area = 0
        for dx in [0, 1]:
            for dy in [0, 1, 2, 3]:
                nx = x + dx
                ny = y + dy
                if f(nx, ny) == 0:
                    area += 1
        return area / 4.0

    total = 0.0
    for x in range(A, C):
        for y in range(B, D):
            total += count(x, y)
    return total * 2

A, B, C, D = map(int, input().split())
print(int(compute_area(A, B, C, D)))
def solve():
    n = int(input())
    rectangles = []
    for _ in range(n):
        rectangles.append(list(map(int, input().split())))

    grid = [[0] * 101 for _ in range(101)]

    for a, b, c, d in rectangles:
        for x in range(a, b):
            for y in range(c, d):
                grid[x][y] = 1

    total_area = 0
    for x in range(101):
        for y in range(101):
            total_area += grid[x][y]

    print(total_area)

solve()
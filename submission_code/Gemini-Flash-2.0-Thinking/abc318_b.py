def solve():
    n = int(input())
    rectangles = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        rectangles.append((a, b, c, d))

    grid = [[0 for _ in range(101)] for _ in range(101)]

    for a, b, c, d in rectangles:
        for y in range(c, d):
            for x in range(a, b):
                grid[y][x] = 1

    count = 0
    for row in grid:
        count += sum(row)

    print(count)

solve()
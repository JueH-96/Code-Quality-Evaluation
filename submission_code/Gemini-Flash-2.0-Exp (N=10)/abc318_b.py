def solve():
    n = int(input())
    rects = []
    for _ in range(n):
        rects.append(list(map(int, input().split())))

    grid = [[0 for _ in range(101)] for _ in range(101)]

    for a, b, c, d in rects:
        for x in range(a, b):
            for y in range(c, d):
                grid[x][y] = 1
    
    area = 0
    for row in grid:
        area += sum(row)
    
    print(area)

solve()
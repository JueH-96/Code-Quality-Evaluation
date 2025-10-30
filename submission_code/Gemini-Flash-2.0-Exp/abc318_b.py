def solve():
    n = int(input())
    rectangles = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        rectangles.append((a, b, c, d))
    
    grid = [[0] * 100 for _ in range(100)]
    
    for a, b, c, d in rectangles:
        for i in range(a, b):
            for j in range(c, d):
                grid[i][j] = 1
    
    area = 0
    for i in range(100):
        for j in range(100):
            if grid[i][j] == 1:
                area += 1
    
    print(area)

solve()
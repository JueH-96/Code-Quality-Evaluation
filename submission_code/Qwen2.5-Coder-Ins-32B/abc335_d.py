# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    
    grid = [[''] * N for _ in range(N)]
    center = N // 2
    grid[center][center] = 'T'
    
    x, y = center, center
    dx, dy = 0, 1
    num = 1
    
    for _ in range(N * N - 1):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < N and 0 <= ny < N) or grid[nx][ny] != '':
            dx, dy = dy, -dx
            nx, ny = x + dx, y + dy
        grid[nx][ny] = str(num)
        num += 1
        x, y = nx, ny
    
    for row in grid:
        print(' '.join(row))

solve()
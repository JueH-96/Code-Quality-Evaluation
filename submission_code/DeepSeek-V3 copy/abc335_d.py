# YOUR CODE HERE
def main():
    N = int(input())
    grid = [[0 for _ in range(N)] for _ in range(N)]
    center = N // 2
    grid[center][center] = 'T'
    
    x, y = 0, 0
    dx, dy = 0, 1
    current = 1
    
    for layer in range(N // 2 + 1):
        for _ in range(4):
            for _ in range(N - 1 - 2 * layer):
                if grid[x][y] == 0:
                    grid[x][y] = current
                    current += 1
                x += dx
                y += dy
            dx, dy = dy, -dx
    
    for row in grid:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()
# YOUR CODE HERE
def main():
    import sys
    N = int(sys.stdin.readline())
    grid = [[0 for _ in range(N)] for _ in range(N)]
    center = N // 2
    grid[center][center] = 'T'
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x, y = 0, 0
    current_dir = 0
    num = 1
    
    for layer in range(N // 2 + 1):
        for _ in range(4):
            for _ in range(2 * layer):
                if x == center and y == center:
                    x += directions[current_dir][0]
                    y += directions[current_dir][1]
                    continue
                grid[x][y] = num
                num += 1
                x += directions[current_dir][0]
                y += directions[current_dir][1]
            current_dir = (current_dir + 1) % 4
    
    for row in grid:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()
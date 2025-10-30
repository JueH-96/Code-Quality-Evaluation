def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    for i in range(1, N // 2 + 1):
        for x in range(i, N + 1 - i + 1):
            y = N + 1 - x
            # Swap colors
            grid[y - 1][i - 1], grid[x - 1][N - i] = grid[x - 1][N - i], grid[y - 1][i - 1]
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()
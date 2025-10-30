def main():
    N = int(input().strip())
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    for i in range(1, N + 1):
        j = N + 1 - i
        if i <= j:
            color = '#' if i % 2 == 1 else '.'
            start = i - 1
            end = j - 1
            for r in range(start, end + 1):
                for c in range(start, end + 1):
                    grid[r][c] = color
                    
    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    main()
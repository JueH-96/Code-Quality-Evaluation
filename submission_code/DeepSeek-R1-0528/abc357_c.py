def main():
    n = int(input().strip())
    grid = ['#']
    for _ in range(n):
        size_prev = len(grid)
        new_grid = []
        for row in grid:
            new_grid.append(row * 3)
        for row in grid:
            new_grid.append(row + '.' * size_prev + row)
        for row in grid:
            new_grid.append(row * 3)
        grid = new_grid
    for line in grid:
        print(line)

if __name__ == "__main__":
    main()
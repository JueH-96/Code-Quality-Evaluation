def draw_carpet(n):
    size = 3 ** n
    grid = [['.' for _ in range(size)] for _ in range(size)]

    def fill_grid(x, y, size):
        if size == 1:
            grid[y][x] = '#'
        else:
            new_size = size // 3
            for dx in range(3):
                for dy in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    fill_grid(x + dx * new_size, y + dy * new_size, new_size)

    fill_grid(0, 0, size)
    return grid


def print_carpet(n):
    grid = draw_carpet(n)
    for row in grid:
        print(''.join(row))


if __name__ == "__main__":
    n = int(input())
    print_carpet(n)
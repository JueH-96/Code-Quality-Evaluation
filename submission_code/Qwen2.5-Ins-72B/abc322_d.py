def rotate_polyomino(polyomino):
    return [''.join(row) for row in zip(*polyomino[::-1])]

def translate_polyomino(polyomino, dx, dy):
    new_polyomino = [['.' for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                ni, nj = i + dx, j + dy
                if 0 <= ni < 4 and 0 <= nj < 4:
                    new_polyomino[ni][nj] = '#'
    return new_polyomino

def is_valid_placement(grid, polyomino, dx, dy):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                ni, nj = i + dx, j + dy
                if not (0 <= ni < 4 and 0 <= nj < 4) or grid[ni][nj] == '#':
                    return False
    return True

def place_polyomino(grid, polyomino, dx, dy):
    new_grid = [row[:] for row in grid]
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                ni, nj = i + dx, j + dy
                if 0 <= ni < 4 and 0 <= nj < 4:
                    new_grid[ni][nj] = '#'
    return new_grid

def can_fill_grid(grid, polyominoes, index=0):
    if index == len(polyominoes):
        for row in grid:
            if '.' in row:
                return False
        return True

    for _ in range(4):
        for dx in range(4):
            for dy in range(4):
                if is_valid_placement(grid, polyominoes[index], dx, dy):
                    new_grid = place_polyomino(grid, polyominoes[index], dx, dy)
                    if can_fill_grid(new_grid, polyominoes, index + 1):
                        return True
        polyominoes[index] = rotate_polyomino(polyominoes[index])
    return False

def main():
    polyominoes = []
    for _ in range(3):
        polyomino = [input().strip() for _ in range(4)]
        polyominoes.append(polyomino)

    grid = [['.' for _ in range(4)] for _ in range(4)]
    if can_fill_grid(grid, polyominoes):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
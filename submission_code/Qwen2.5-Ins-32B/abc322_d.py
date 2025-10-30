import itertools

def rotate(p):
    return [''.join(x) for x in zip(*p[::-1])]

def all_rotations(p):
    for _ in range(4):
        yield p
        p = rotate(p)

def can_fill(grid, piece, x, y):
    for i in range(4):
        for j in range(4):
            if piece[i][j] == '#' and grid[x+i][y+j] == '#':
                return False
    return True

def place_piece(grid, piece, x, y, place):
    new_grid = [row[:] for row in grid]
    for i in range(4):
        for j in range(4):
            if piece[i][j] == '#':
                if place:
                    new_grid[x+i][y+j] = '#'
                else:
                    new_grid[x+i][y+j] = '.'
    return new_grid

def solve(grid, pieces, index=0):
    if index == 3:
        return all(cell == '#' for row in grid for cell in row)
    for piece in pieces[index]:
        for x in range(2):
            for y in range(2):
                if can_fill(grid, piece, x, y):
                    new_grid = place_piece(grid, piece, x, y, True)
                    if solve(new_grid, pieces, index + 1):
                        return True
                    place_piece(new_grid, piece, x, y, False)
    return False

def main():
    pieces = [[input() for _ in range(4)] for _ in range(3)]
    all_pieces = [[list(all_rotations(piece)) for piece in pieces]]
    grid = [['.' for _ in range(4)] for _ in range(4)]
    if solve(grid, all_pieces):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
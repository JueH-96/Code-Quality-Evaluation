from itertools import permutations

def parse_polyominoes(input_lines):
    polyominoes = []
    for i in range(0, 16, 4):
        polyomino = [input_lines[j][i:i+4] for j in range(4)]
        polyominoes.append(polyomino)
    return polyominoes

def rotate_polyomino(polyomino):
    return [''.join(polyomino[3-i][j] for i in range(4)) for j in range(4)]

def place_polyomino(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                if grid[x+i][y+j] == '#':
                    return False
                grid[x+i][y+j] = '#'
    return True

def remove_polyomino(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x+i][y+j] = '.'

def can_place_polyominoes(polyominoes):
    grid = [['.' for _ in range(4)] for _ in range(4)]
    for perm in permutations(polyominoes):
        for rot in range(4):
            rotated_polyomino = rotate_polyomino(perm[0])
            for x in range(5):
                for y in range(5):
                    if place_polyomino(grid, rotated_polyomino, x, y):
                        if can_place_polyominoes(perm[1:]):
                            return True
                        remove_polyomino(grid, rotated_polyomino, x, y)
    return False

def main():
    import sys
    input = sys.stdin.read
    input_lines = input().strip().split()
    polyominoes = parse_polyominoes(input_lines)
    if can_place_polyominoes(polyominoes):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
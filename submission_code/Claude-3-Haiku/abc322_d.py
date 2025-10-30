from typing import List

def can_fill_grid(polyominoes: List[List[List[str]]]) -> bool:
    # Helper function to check if a polyomino can be placed at a given position
    def can_place(grid, polyomino, row, col):
        for i in range(4):
            for j in range(4):
                if polyomino[i][j] == '#':
                    if row + i >= 4 or col + j >= 4 or grid[row + i][col + j] == '#':
                        return False
        return True

    # Helper function to place a polyomino on the grid
    def place_polyomino(grid, polyomino, row, col):
        for i in range(4):
            for j in range(4):
                if polyomino[i][j] == '#':
                    grid[row + i][col + j] = '#'

    # Helper function to remove a polyomino from the grid
    def remove_polyomino(grid, polyomino, row, col):
        for i in range(4):
            for j in range(4):
                if polyomino[i][j] == '#':
                    grid[row + i][col + j] = '.'

    # Main function to solve the problem
    grid = [['.' for _ in range(4)] for _ in range(4)]

    for i in range(3):
        for row in range(4):
            for col in range(4):
                if can_place(grid, polyominoes[i], row, col):
                    place_polyomino(grid, polyominoes[i], row, col)
                    if all(grid[j][k] == '#' for j in range(4) for k in range(4)):
                        return "Yes"
                    remove_polyomino(grid, polyominoes[i], row, col)

    return "No"

# Read the input
polyominoes = []
for _ in range(12):
    row = input().strip()
    polyomino = [list(row[i:i+4]) for i in range(0, 16, 4)]
    polyominoes.append(polyomino)

# Solve the problem
print(can_fill_grid(polyominoes))
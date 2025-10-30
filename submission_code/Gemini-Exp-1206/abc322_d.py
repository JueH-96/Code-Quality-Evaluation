def rotate(piece):
    return list(zip(*piece[::-1]))

def solve():
    pieces = []
    for _ in range(3):
        piece = []
        for _ in range(4):
            piece.append(list(input()))
        pieces.append(piece)

    def get_shapes(piece):
        shapes = []
        for _ in range(4):
            min_r, min_c = 4, 4
            max_r, max_c = -1, -1
            for r in range(4):
                for c in range(4):
                    if piece[r][c] == '#':
                        min_r = min(min_r, r)
                        min_c = min(min_c, c)
                        max_r = max(max_r, r)
                        max_c = max(max_c, c)
            
            shape = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    row.append(piece[r][c])
                shape.append(row)
            
            if shape not in shapes:
                shapes.append(shape)
            piece = rotate(piece)
        return shapes

    all_shapes = [get_shapes(p) for p in pieces]

    def backtrack(grid, shape_indices):
        if len(shape_indices) == 3:
            for r in range(4):
                for c in range(4):
                    if grid[r][c] == '.':
                        return False
            return True

        piece_index = len(shape_indices)
        for shape in all_shapes[piece_index]:
            for r_start in range(5 - len(shape)):
                for c_start in range(5 - len(shape[0])):
                    
                    new_grid = [row[:] for row in grid]
                    valid_placement = True
                    for r in range(len(shape)):
                        for c in range(len(shape[0])):
                            if shape[r][c] == '#':
                                if new_grid[r_start + r][c_start + c] == '#':
                                    valid_placement = False
                                    break
                                new_grid[r_start + r][c_start + c] = '#'
                        if not valid_placement:
                            break
                    
                    if valid_placement:
                        if backtrack(new_grid, shape_indices + [shape]):
                            return True
        return False

    initial_grid = [['.' for _ in range(4)] for _ in range(4)]
    if backtrack(initial_grid, []):
        print("Yes")
    else:
        print("No")

solve()
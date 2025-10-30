import sys

def solve():
    polyomino_grids = []
    for _ in range(3):
        grid = [input() for _ in range(4)]
        polyomino_grids.append(grid)
    
    polyomino_shapes = []
    for grid in polyomino_grids:
        shape_coords = []
        min_r, min_c = 4, 4
        found_first = False
        for r in range(4):
            for c in range(4):
                if grid[r][c] == '#':
                    if not found_first:
                        min_r, min_c = r, c
                        found_first = True
                    shape_coords.append((r, c))
        normalized_shape = []
        for r, c in shape_coords:
            normalized_shape.append((r - min_r, c - min_c))
        polyomino_shapes.append(normalized_shape)
        
    rotated_shapes_list = []
    for shape in polyomino_shapes:
        rotations = []
        current_shape = set(shape)
        for _ in range(4):
            rotations.append(list(current_shape))
            rotated_shape = set()
            min_r_rot, min_c_rot = 4, 4
            for r, c in current_shape:
                rotated_point = (c, -r)
                rotated_shape.add(rotated_point)
                min_r_rot = min(min_r_rot, rotated_point[0])
                min_c_rot = min(min_c_rot, rotated_point[1])
            normalized_rotated_shape = set()
            for r, c in rotated_shape:
                normalized_rotated_shape.add((r - min_r_rot, c - min_c_rot))
            current_shape = normalized_rotated_shape
        unique_rotations = []
        seen_shapes = set()
        for rotation in rotations:
            rotation_tuple = tuple(sorted(rotation))
            if rotation_tuple not in seen_shapes:
                unique_rotations.append(rotation)
                seen_shapes.add(rotation_tuple)
        rotated_shapes_list.append(unique_rotations)

    polyomino_sizes = [len(shape) for shape in polyomino_shapes]
    if sum(polyomino_sizes) != 16:
        print("No")
        return
        
    def is_valid_placement(grid, shape, start_row, start_col):
        positions = []
        for r_rel, c_rel in shape:
            r_abs, c_abs = start_row + r_rel, start_col + c_rel
            if not (0 <= r_abs < 4 and 0 <= c_abs < 4):
                return False
            if grid[r_abs][c_abs] != 0:
                return False
            positions.append((r_abs, c_abs))
        return positions

    def solve_recursive(polyomino_index, current_grid):
        if polyomino_index == 3:
            for r in range(4):
                for c in range(4):
                    if current_grid[r][c] == 0:
                        return False
            return True
            
        for shape in rotated_shapes_list[polyomino_index]:
            for start_row in range(4):
                for start_col in range(4):
                    positions = is_valid_placement(current_grid, shape, start_row, start_col)
                    if positions:
                        next_grid = [list(row) for row in current_grid]
                        for r_abs, c_abs in positions:
                            next_grid[r_abs][c_abs] = polyomino_index + 1
                        if solve_recursive(polyomino_index + 1, next_grid):
                            return True
        return False

    initial_grid = [[0 for _ in range(4)] for _ in range(4)]
    if solve_recursive(0, initial_grid):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()
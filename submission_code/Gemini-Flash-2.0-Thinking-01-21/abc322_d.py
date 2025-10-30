import copy

def get_polyomino_shape(polyomino_grid):
    shape_coords = []
    for r in range(4):
        for c in range(4):
            if polyomino_grid[r][c] == '#':
                shape_coords.append((r, c))
    if not shape_coords:
        return []
    min_r = min(r for r, c in shape_coords)
    min_c = min(c for r, c in shape_coords)
    normalized_coords = []
    for r, c in shape_coords:
        normalized_coords.append((r - min_r, c - min_c))
    return normalized_coords

def rotate_polyomino(shape_coords):
    if not shape_coords:
        return []
    max_r = max(r for r, c in shape_coords) if shape_coords else 0
    rotated_coords = []
    for r, c in shape_coords:
        rotated_coords.append((c, max_r - r))
    min_r = min(r for r, c in rotated_coords) if rotated_coords else 0
    min_c = min(c for r, c in rotated_coords) if rotated_coords else 0
    normalized_rotated_coords = []
    for r, c in rotated_coords:
        normalized_rotated_coords.append((r - min_r, c - min_c))
    return normalized_rotated_coords

def get_all_rotations(shape_coords):
    rotations = []
    current_shape = shape_coords
    for _ in range(4):
        rotations.append(current_shape)
        current_shape = rotate_polyomino(current_shape)
    unique_rotations = []
    seen_shapes = set()
    for shape in rotations:
        shape_tuple = tuple(sorted(shape))
        if shape_tuple not in seen_shapes:
            unique_rotations.append(shape)
            seen_shapes.add(shape_tuple)
    return unique_rotations

polyomino_grids = []
for _ in range(3):
    grid = []
    for _ in range(4):
        grid.append(input())
    polyomino_grids.append(grid)

polyomino_shapes = []
total_squares = 0
for grid in polyomino_grids:
    shape = get_polyomino_shape(grid)
    polyomino_shapes.append(shape)
    total_squares += len(shape)

if total_squares != 16:
    print("No")
else:
    rotation_sets = []
    for shape in polyomino_shapes:
        rotation_sets.append(get_all_rotations(shape))

    def is_valid_placement(grid_state, shape, start_row, start_col):
        for r_offset, c_offset in shape:
            row, col = start_row + r_offset, start_col + c_offset
            if not (0 <= row < 4 and 0 <= col < 4 and grid_state[row][col] == 0):
                return False
        return True

    def place_polyomino(grid_state, shape, start_row, start_col, polyomino_index):
        new_grid_state = [list(row) for row in grid_state]
        for r_offset, c_offset in shape:
            row, col = start_row + r_offset, start_col + c_offset
            new_grid_state[row][col] = polyomino_index
        return tuple("".join(map(str,row)) for row in new_grid_state)

    def solve(polyomino_index, current_grid_state):
        if polyomino_index == 3:
            return True
        
        for rotation_shape in rotation_sets[polyomino_index]:
            max_r = max((r for r,c in rotation_shape), default=-1)
            max_c = max((c for r,c in rotation_shape), default=-1)
            shape_height = max_r + 1
            shape_width = max_c + 1
            for start_row in range(4 - shape_height + 1):
                for start_col in range(4 - shape_width + 1):
                    if is_valid_placement(current_grid_state, rotation_shape, start_row, start_col):
                        next_grid_state = place_polyomino(current_grid_state, rotation_shape, start_row, start_col, polyomino_index + 1)
                        if solve(polyomino_index + 1, next_grid_state):
                            return True
        return False

    initial_grid = tuple(["0000"] * 4)
    if solve(0, initial_grid):
        print("Yes")
    else:
        print("No")
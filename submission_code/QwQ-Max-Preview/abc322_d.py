def read_polyomino():
    return [input().strip() for _ in range(4)]

# Read the three polyominoes
polys = [read_polyomino() for _ in range(3)]

# Extract cells for each polyomino
cells_list = []
for p in polys:
    cells = []
    for j in range(4):
        for k in range(4):
            if p[j][k] == '#':
                cells.append((j, k))
    cells_list.append(cells)

# Check if the total number of cells is 16
total = sum(len(cells) for cells in cells_list)
if total != 16:
    print("No")
    exit()

# Function to normalize and generate rotations
def generate_masks(cells):
    # Normalize the cells
    min_j = min(j for j, k in cells)
    min_k = min(k for j, k in cells)
    normalized = [(j - min_j, k - min_k) for j, k in cells]
    
    # Generate all unique rotated shapes
    rotated_shapes = set()
    for rot in range(4):
        rotated = []
        for x, y in normalized:
            if rot == 0:
                new_x, new_y = x, y
            elif rot == 1:  # 90 degrees
                new_x, new_y = y, -x
            elif rot == 2:  # 180 degrees
                new_x, new_y = -x, -y
            elif rot == 3:  # 270 degrees
                new_x, new_y = -y, x
            rotated.append((new_x, new_y))
        # Normalize the rotated shape
        min_x = min(p[0] for p in rotated)
        min_y = min(p[1] for p in rotated)
        normalized_rot = [(p[0] - min_x, p[1] - min_y) for p in rotated]
        shape = frozenset(normalized_rot)
        rotated_shapes.add(shape)
    
    # Generate all possible masks for each rotated shape
    masks = set()
    for shape in rotated_shapes:
        shape_list = list(shape)
        if not shape_list:
            continue
        max_x = max(p[0] for p in shape_list)
        max_y = max(p[1] for p in shape_list)
        dx_max = 4 - max_x
        dy_max = 4 - max_y
        for dx in range(dx_max):
            for dy in range(dy_max):
                mask = 0
                for x, y in shape_list:
                    tx = x + dx
                    ty = y + dy
                    if 0 <= tx < 4 and 0 <= ty < 4:
                        mask |= 1 << (tx * 4 + ty)
                masks.add(mask)
    return masks

# Generate masks for each polyomino
masks_list = []
for cells in cells_list:
    masks = generate_masks(cells)
    masks_list.append(masks)

# Check all combinations of masks from the three polyominoes
def check_combinations(masks1, masks2, masks3):
    for m1 in masks1:
        for m2 in masks2:
            if (m1 & m2) != 0:
                continue
            combined = m1 | m2
            for m3 in masks3:
                if (combined & m3) == 0 and (combined | m3) == 0xffff:
                    return True
    return False

if check_combinations(masks_list[0], masks_list[1], masks_list[2]):
    print("Yes")
else:
    print("No")
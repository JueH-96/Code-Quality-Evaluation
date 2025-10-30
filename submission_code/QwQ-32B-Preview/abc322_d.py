def get_offsets(polyomino):
    offsets = []
    for row in range(4):
        for col in range(4):
            if polyomino[row][col] == '#':
                offsets.append((row, col))
    return offsets

def rotate_offsets(offsets, rotation):
    if rotation == 0:
        return offsets
    elif rotation == 1:
        return [(col, -row) for row, col in offsets]
    elif rotation == 2:
        return [(-row, -col) for row, col in offsets]
    elif rotation == 3:
        return [(-col, row) for row, col in offsets]

def generate_placements(polyomino):
    placements = []
    offsets = get_offsets(polyomino)
    for rotation in range(4):
        rotated_offsets = rotate_offsets(offsets, rotation)
        min_row = min(o[0] for o in rotated_offsets)
        min_col = min(o[1] for o in rotated_offsets)
        adjusted_offsets = [(o[0] - min_row, o[1] - min_col) for o in rotated_offsets]
        max_poly_row = max(o[0] for o in adjusted_offsets)
        max_poly_col = max(o[1] for o in adjusted_offsets)
        for dx in range(4 - max_poly_row):
            for dy in range(4 - max_poly_col):
                placement = set((dx + o[0], dy + o[1]) for o in adjusted_offsets)
                placements.append(placement)
    return placements

def backtrack(occupied, placements, index):
    if index == len(placements):
        return len(occupied) == 16
    for placement in placements[index]:
        if placement.isdisjoint(occupied):
            new_occupied = occupied.union(placement)
            if backtrack(new_occupied, placements, index + 1):
                return True
    return False

def main():
    input_lines = [input().strip() for _ in range(12)]
    polyominoes = [input_lines[i*4:(i+1)*4] for i in range(3)]
    placements = [generate_placements(p) for p in polyominoes]
    if backtrack(set(), placements, 0):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
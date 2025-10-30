# YOUR CODE HERE
import sys

def read_polyomino(lines):
    shape = []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == '#':
                shape.append((r, c))
    return shape

def rotate(shape):
    return [(-c, r) for r, c in shape]

def normalize(shape):
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return sorted([(r - min_r, c - min_c) for r, c in shape])

def get_unique_rotations(shape):
    rotations = []
    current = shape
    for _ in range(4):
        current = rotate(current)
        norm = normalize(current)
        if norm not in rotations:
            rotations.append(norm)
    return rotations

def get_all_placements(rotations):
    placements = []
    for rot in rotations:
        max_r = max(r for r, c in rot)
        max_c = max(c for r, c in rot)
        for dr in range(4 - max_r):
            for dc in range(4 - max_c):
                cells = []
                mask = 0
                for r, c in rot:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        pos = nr * 4 + nc
                        mask |= (1 << pos)
                    else:
                        break
                else:
                    placements.append(mask)
    return placements

def main():
    input_lines = [sys.stdin.readline().strip() for _ in range(12)]
    polyominoes = []
    for i in range(3):
        poly = read_polyomino(input_lines[i*4:(i+1)*4])
        polyominoes.append(poly)

    all_rotations = [get_unique_rotations(p) for p in polyominoes]
    all_placements = [get_all_placements(r) for r in all_rotations]

    # Sort polyominoes by number of placements to optimize backtracking
    order = sorted(range(3), key=lambda x: len(all_placements[x]))
    sorted_placements = [all_placements[i] for i in order]

    def backtrack(i, used):
        if i == 3:
            return used == 0xFFFF
        for placement in sorted_placements[i]:
            if (placement & used) == 0:
                if backtrack(i+1, used | placement):
                    return True
        return False

    if backtrack(0, 0):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
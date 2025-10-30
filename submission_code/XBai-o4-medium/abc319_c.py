import itertools

def main():
    grid = [list(map(int, input().split())) for _ in range(3)]
    
    all_lines = []
    # Rows
    for i in range(3):
        all_lines.append([(i, 0), (i, 1), (i, 2)])
    # Columns
    for j in range(3):
        all_lines.append([(0, j), (1, j), (2, j)])
    # Diagonals
    all_lines.append([(0, 0), (1, 1), (2, 2)])
    all_lines.append([(2, 0), (1, 1), (0, 2)])
    
    all_cells = [(i, j) for i in range(3) for j in range(3)]
    
    valid_count = 0
    
    # Precompute 9! once
    total_permutations = 362880  # 9!
    
    for perm in itertools.permutations(all_cells):
        pos_dict = {cell: idx for idx, cell in enumerate(perm)}
        values_in_perm = [grid[i][j] for i, j in perm]
        valid = True
        for line in all_lines:
            cells = line
            line_positions = [pos_dict[cell] for cell in cells]
            sorted_positions = sorted(line_positions)
            v0 = values_in_perm[sorted_positions[0]]
            v1 = values_in_perm[sorted_positions[1]]
            v2 = values_in_perm[sorted_positions[2]]
            if v0 == v1 and v0 != v2:
                valid = False
                break
        if valid:
            valid_count += 1
    
    probability = valid_count / total_permutations
    # Print with sufficient decimal precision
    print("{0:.15f}".format(probability))

if __name__ == "__main__":
    main()
import sys
import itertools
import math

def solve():
    """
    Reads the grid, finds the probability of not getting disappointed, and prints it.
    """
    # Read the 3x3 grid from standard input.
    # We use 0-based indexing for rows and columns (0 to 2).
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]

    # Define the coordinates for all 8 lines (3 rows, 3 columns, 2 diagonals).
    line_coords = [
        # Rows
        [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
        # Columns
        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
        # Diagonals
        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
    ]

    # A line is "gari" if it has two cells with the same number and one different.
    # We store the coordinates of the two same-valued cells and the different-valued cell.
    gari_lines = []
    for line in line_coords:
        c1, c2, c3 = line
        v1 = grid[c1[0]][c1[1]]
        v2 = grid[c2[0]][c2[1]]
        v3 = grid[c3[0]][c3[1]]

        # A line is gari if its values have exactly 2 unique numbers.
        # The problem guarantees no line has 3 same numbers.
        if len(set([v1, v2, v3])) == 2:
            if v1 == v2:
                gari_lines.append(((c1, c2), c3))
            elif v1 == v3:
                gari_lines.append(((c1, c3), c2))
            else:  # v2 == v3
                gari_lines.append(((c2, c3), c1))

    # If there are no gari lines, disappointment is impossible. Probability is 1.
    if not gari_lines:
        print(1.0)
        return

    all_cells = [(r, c) for r in range(3) for c in range(3)]
    total_permutations = math.factorial(9)
    good_permutations_count = 0

    # Iterate through all 9! permutations of cell visiting orders.
    for p in itertools.permutations(all_cells):
        is_disappointed = False
        
        # Create a map for quick lookup of a cell's position in the current order.
        pos_map = {cell: i for i, cell in enumerate(p)}

        # Check each gari line for the disappointment condition.
        for same_cells, diff_cell in gari_lines:
            pos_diff = pos_map[diff_cell]
            pos_same1 = pos_map[same_cells[0]]
            pos_same2 = pos_map[same_cells[1]]

            # Disappointment occurs if the two same-valued cells are seen before the different-valued one.
            if pos_same1 < pos_diff and pos_same2 < pos_diff:
                is_disappointed = True
                break  # This permutation is bad; no need to check further.
        
        if not is_disappointed:
            good_permutations_count += 1
            
    # Probability is the ratio of good permutations to the total.
    probability = good_permutations_count / total_permutations
    
    print(f"{probability:.15f}")

solve()
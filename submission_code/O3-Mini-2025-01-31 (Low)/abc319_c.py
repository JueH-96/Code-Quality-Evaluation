def main():
    import sys
    from itertools import permutations
    import math

    data = sys.stdin.read().split()
    # There should be 9 numbers (forming a 3x3 grid).
    if len(data) < 9:
        return
    # Build a 1D representation of the grid.
    grid = list(map(int, data))  # cell index: 0 to 8; row i, column j as: index = 3*i + j
    
    # Pre-define all eight lines (three rows, three columns, two diagonals)
    lines = []
    # Rows:
    for r in range(3):
        lines.append([r*3, r*3+1, r*3+2])
    # Columns:
    for c in range(3):
        lines.append([c, c+3, c+6])
    # Diagonals:
    lines.append([0, 4, 8])
    lines.append([2, 4, 6])
    
    total = math.factorial(9)
    valid = 0  # count of orders that allow seeing all numbers without disappointment

    # Iterate over every possible order in which Takahashi sees the nine cells.
    # A permutation order is a tuple with indices 0..8 representing the order of revelation.
    for order in permutations(range(9)):
        # Create a list to quickly lookup the time (position in order) a given cell is seen.
        pos_order = [0] * 9
        for t, pos in enumerate(order):
            pos_order[pos] = t
        
        disappointed = False
        # Check every line.
        for line in lines:
            # Get the positions in this line, sorted by the order in which they were seen.
            sorted_cells = sorted(line, key=lambda pos: pos_order[pos])
            first, second, third = grid[sorted_cells[0]], grid[sorted_cells[1]], grid[sorted_cells[2]]
            # Takahashi gets disappointed for this line if:
            # The first two numbers seen (in that line) are equal,
            # but the third one is different.
            if first == second and third != first:
                disappointed = True
                break
        if not disappointed:
            valid += 1

    probability = valid / total
    sys.stdout.write(str(probability))


if __name__ == '__main__':
    main()
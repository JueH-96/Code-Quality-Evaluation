import itertools

def is_disappointing(perm, grid, lines):
    """Check if a permutation is disappointing for Takahashi."""
    for line in lines:
        a, b, c = line
        # Get the order in which Takahashi sees cells a, b, c
        seen_order = sorted([a, b, c], key=lambda x: list(perm).index(x))
        # Check if the first two cells seen have the same value, and the third has a different one.
        if grid[seen_order[0]] == grid[seen_order[1]] != grid[seen_order[2]]:
            return True
    return False

def solve():
    # Read the grid
    grid = []
    for i in range(3):
        row = list(map(int, input().split()))
        grid.extend(row)
    
    # Define the lines (represent a cell as its index in the 1D grid)
    lines = [
        # Rows
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        # Columns
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        # Diagonals
        [0, 4, 8], [2, 4, 6]
    ]
    
    # Generate all permutations of the 9 cells
    all_perms = list(itertools.permutations(range(9)))
    
    # Count non-disappointing permutations
    non_disappointing = 0
    for perm in all_perms:
        if not is_disappointing(perm, grid, lines):
            non_disappointing += 1
    
    # Compute the probability
    return non_disappointing / len(all_perms)

# Main
print(solve())
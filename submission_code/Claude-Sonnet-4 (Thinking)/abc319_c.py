import itertools

def solve():
    # Read the grid
    grid = []
    for i in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Define all lines
    lines = []
    # Horizontal lines
    for i in range(3):
        lines.append([(i, 0), (i, 1), (i, 2)])
    # Vertical lines
    for j in range(3):
        lines.append([(0, j), (1, j), (2, j)])
    # Diagonal lines
    lines.append([(0, 0), (1, 1), (2, 2)])
    lines.append([(2, 0), (1, 1), (0, 2)])
    
    # Generate all positions
    positions = [(i, j) for i in range(3) for j in range(3)]
    
    successful_permutations = 0
    total_permutations = 0
    
    # Try all permutations of the 9 positions
    for perm in itertools.permutations(positions):
        total_permutations += 1
        disappointed = False
        
        # Create a mapping from position to the order it's seen
        pos_to_order = {pos: i for i, pos in enumerate(perm)}
        
        # Check each line for disappointment
        for line in lines:
            # Sort the positions in this line by the order they're seen
            line_positions_with_order = [(pos_to_order[pos], pos) for pos in line]
            line_positions_with_order.sort()
            
            # Get the values at these positions in the order they're seen
            values_in_order = [grid[pos[0]][pos[1]] for _, pos in line_positions_with_order]
            
            # Check if the first two values are the same and the third is different
            if values_in_order[0] == values_in_order[1] and values_in_order[1] != values_in_order[2]:
                disappointed = True
                break
        
        if not disappointed:
            successful_permutations += 1
    
    probability = successful_permutations / total_permutations
    print(probability)

solve()
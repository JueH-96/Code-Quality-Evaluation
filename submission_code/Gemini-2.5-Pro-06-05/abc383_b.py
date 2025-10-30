import sys
import itertools

def main():
    """
    Main function to read input, solve the problem, and print the output.
    """
    # Read problem parameters from standard input
    try:
        H, W, D = map(int, sys.stdin.readline().split())
        S = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle cases with no input
        return

    # Identify all floor cells ('.') using a list comprehension
    floor_cells = [(r, c) for r in range(H) for c in range(W) if S[r][c] == '.']
    
    max_humidified_count = 0

    # The problem guarantees at least two floor cells.
    # We iterate through all distinct pairs of floor cells for humidifier placement.
    # itertools.combinations is an efficient and Pythonic way to generate these pairs.
    for h1, h2 in itertools.combinations(floor_cells, 2):
        # h1 and h2 are the (row, col) coordinates of the two humidifiers.
        
        current_humidified_count = 0
        
        # For the chosen pair of humidifiers, count how many floor cells are humidified.
        # A floor cell is humidified if its Manhattan distance to at least one
        # humidifier is less than or equal to D.
        for f_cell in floor_cells:
            # Calculate Manhattan distances
            dist1 = abs(f_cell[0] - h1[0]) + abs(f_cell[1] - h1[1])
            dist2 = abs(f_cell[0] - h2[0]) + abs(f_cell[1] - h2[1])

            # Check the condition from the problem statement
            if dist1 <= D or dist2 <= D:
                current_humidified_count += 1
        
        # Update the overall maximum count found so far
        max_humidified_count = max(max_humidified_count, current_humidified_count)
            
    # Print the final result to standard output
    print(max_humidified_count)


if __name__ == "__main__":
    main()
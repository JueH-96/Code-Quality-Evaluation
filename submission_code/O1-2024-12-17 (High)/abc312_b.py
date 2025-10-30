def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    grid = data[2:2+N]  # Each of these N elements is a string representing a row
    
    # Define the coordinates for the top-left 3×3 black region
    top_left_black = [(r, c) for r in range(3) for c in range(3)]
    # Define the coordinates for the cells adjacent to the top-left 3×3 region (must be white)
    top_left_adj = [(r, c) for r in range(4) for c in range(4)
                    if not (r < 3 and c < 3)]  # exclude the black region itself

    # Define the coordinates for the bottom-right 3×3 black region
    bottom_right_black = [(r, c) for r in range(6, 9) for c in range(6, 9)]
    # Define the coordinates for the cells adjacent to the bottom-right 3×3 region (must be white)
    bottom_right_adj = [(r, c) for r in range(5, 9) for c in range(5, 9)
                        if not (r >= 6 and c >= 6)]
    
    # Function to check if a 9×9 subgrid with top-left corner at (sr, sc) is a valid TaK Code
    def is_tak_code(sr, sc):
        # Check the top-left black 3×3
        for r, c in top_left_black:
            if grid[sr + r][sc + c] != '#':
                return False
        
        # Check the bottom-right black 3×3
        for r, c in bottom_right_black:
            if grid[sr + r][sc + c] != '#':
                return False
        
        # Check the white adjacency around the top-left 3×3
        for r, c in top_left_adj:
            if grid[sr + r][sc + c] != '.':
                return False
        
        # Check the white adjacency around the bottom-right 3×3
        for r, c in bottom_right_adj:
            if grid[sr + r][sc + c] != '.':
                return False
        
        # If all conditions are satisfied
        return True

    # We will search all possible 9×9 subgrids
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(i, j):
                # Output is 1-based in the problem statement
                results.append((i + 1, j + 1))
    
    # Print results in ascending lexicographical order (already ensured by loops)
    for row, col in results:
        print(row, col)

# Don't forget to call main() at the end
if __name__ == "__main__":
    main()
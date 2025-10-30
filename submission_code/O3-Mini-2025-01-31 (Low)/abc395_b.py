def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    # Initialize grid with a dummy, but we'll fill in via operations.
    grid = [['?' for _ in range(N)] for _ in range(N)]
    
    # Process operations for i = 1 to N
    for i in range(1, N+1):
        j = N + 1 - i
        if i <= j:
            # Determine color: black if i odd, white if i even.
            color = '#' if i % 2 == 1 else '.'
            # Fill the rectangular region from (i, i) to (j, j) 
            # Remember to convert 1-index to 0-index.
            for r in range(i-1, j):
                for c in range(i-1, j):
                    grid[r][c] = color
    
    # Output the grid rows.
    for row in grid:
        print("".join(row))
    
if __name__ == '__main__':
    main()
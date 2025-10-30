def fill_pattern(N):
    # Initialize grid with None (uncolored)
    grid = [[None for _ in range(N)] for _ in range(N)]
    
    # Process each i from 1 to N
    for i in range(1, N+1):
        j = N + 1 - i  # Calculate j
        
        # Only process if i <= j
        if i <= j:
            # Convert to 0-based indexing
            i_0 = i - 1
            j_0 = j - 1
            
            # Determine color based on i being odd or even
            color = '#' if i % 2 == 1 else '.'
            
            # Fill the rectangular region
            for row in range(i_0, j_0 + 1):
                for col in range(i_0, j_0 + 1):
                    grid[row][col] = color

    # Return the final grid
    return grid

def main():
    # Read input
    N = int(input())
    
    # Generate pattern
    pattern = fill_pattern(N)
    
    # Print pattern
    for row in pattern:
        print(''.join(row))

if __name__ == "__main__":
    main()
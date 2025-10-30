def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = data[1:]
    
    # Convert grid to a list of lists for easier manipulation
    grid = [list(row) for row in grid]
    
    # Perform the operations
    for i in range(N // 2):
        # i goes from 0 to N//2-1, which corresponds to 1 to N//2 in 1-based index
        for x in range(i, N - i):
            for y in range(i, N - i):
                # (x, y) -> (y, N-x-1)
                # We need to store the original (x, y) value
                original_color = grid[x][y]
                # Replace the color of cell (y, N-x-1) with the color of cell (x, y)
                grid[y][N - x - 1] = original_color
    
    # Print the final grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()
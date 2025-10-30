def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # Initialize an N x N grid (we will fill it layer by layer)
    grid = [[' '] * N for _ in range(N)]
    
    # The operations are applied for i=1,2,... until i <= N+1-i.
    # In 1-indexed terms, that is for i from 1 to (N+1)//2 (integer division).
    layers = (N + 1) // 2
    for i in range(1, layers + 1):
        # Determine the color for this operation:
        # Black ('#') if i is odd, White ('.') if i is even.
        fill_char = '#' if (i % 2 == 1) else '.'
        
        # In 1-indexed notation, we fill from cell (i, i) to (N+1-i, N+1-i).
        # Convert these coordinates to 0-indexed:
        top = i - 1         # starting row index
        left = i - 1        # starting column index
        bottom = N - i      # ending row index (since (N+1-i)-1 = N-i)
        right = N - i       # ending column index
        
        # Fill the appropriate rectangular region.
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                grid[r][c] = fill_char

    # Build and output the final grid.
    output_lines = [''.join(row) for row in grid]
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()
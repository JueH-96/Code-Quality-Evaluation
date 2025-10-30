def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # Initialize grid with dummy characters (they will be overwritten)
    grid = [[''] * N for _ in range(N)]
    
    # For each i from 1 to N (1-indexed), perform operation when i <= N + 1 - i
    for i in range(1, N + 1):
        j = N + 1 - i
        if i <= j:
            # Determine symbol (black '#' if i is odd, white '.' if even)
            symbol = '#' if i % 2 == 1 else '.'
            # Convert coordinates to zero-indexed:
            # top-left (i, i) becomes (i-1, i-1)
            # bottom-right (j, j) becomes (j-1, j-1)
            for r in range(i - 1, j):
                for c in range(i - 1, j):
                    grid[r][c] = symbol
        # When i > j, do nothing
    
    # Output the grid
    out_lines = [''.join(row) for row in grid]
    sys.stdout.write('
'.join(out_lines))
    
if __name__ == '__main__':
    main()
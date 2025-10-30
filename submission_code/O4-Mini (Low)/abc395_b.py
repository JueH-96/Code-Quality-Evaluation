import sys

def main():
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # Initialize empty grid
    grid = [['?' for _ in range(N)] for _ in range(N)]

    # Apply operations for i = 1 to N
    for i in range(1, N + 1):
        j = N + 1 - i
        if i > j:
            continue
        # Determine fill character: '#' if i is odd, '.' if even
        fill_char = '#' if (i % 2 == 1) else '.'
        # Convert to 0-based indices: rows/cols from i-1 to j-1 inclusive
        row_start = i - 1
        row_end_exclusive = j  # because j-1 inclusive, so stop at j
        for r in range(row_start, row_end_exclusive):
            for c in range(row_start, row_end_exclusive):
                grid[r][c] = fill_char

    # Output the final grid
    out = sys.stdout
    for row in grid:
        out.write(''.join(row) + '
')

if __name__ == "__main__":
    main()
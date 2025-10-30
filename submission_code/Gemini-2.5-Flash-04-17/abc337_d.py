import sys

# Read H, W, K
H, W, K = map(int, sys.stdin.readline().split())

# Read the grid
grid = [sys.stdin.readline().strip() for _ in range(H)]

min_ops = float('inf')

# Check horizontal segments
# A horizontal segment of length K starting at (r, c) (0-based) includes cells (r, c), ..., (r, c+K-1).
# Possible starting columns c range from 0 to W-K.
if W >= K:
    for r in range(H):
        # Initialize window for the first segment (r, 0) to (r, K-1)
        dot_count = 0
        x_count = 0
        for c in range(K):
            if grid[r][c] == '.':
                dot_count += 1
            elif grid[r][c] == 'x':
                x_count += 1

        # Check the first window (starting column 0)
        if x_count == 0:
            min_ops = min(min_ops, dot_count)

        # Slide window to check segments starting from column 1 up to W-K
        # The starting column of the window is `start_c`.
        # When we move from starting column `start_c - 1` to `start_c`,
        # the cell at (r, start_c - 1) leaves and cell at (r, start_c + K - 1) enters.
        for start_c in range(1, W - K + 1):
            # Cell leaving: (r, start_c - 1)
            if grid[r][start_c - 1] == '.':
                dot_count -= 1
            elif grid[r][start_c - 1] == 'x':
                x_count -= 1

            # Cell entering: (r, start_c + K - 1)
            if grid[r][start_c + K - 1] == '.':
                dot_count += 1
            elif grid[r][start_c + K - 1] == 'x':
                x_count += 1

            # Check the current window starting at start_c
            if x_count == 0:
                min_ops = min(min_ops, dot_count)

# Check vertical segments
# A vertical segment of length K starting at (r, c) (0-based) includes cells (r, c), ..., (r+K-1, c).
# Possible starting rows r range from 0 to H-K.
if H >= K:
    for c in range(W):
        # Initialize window for the first segment (0, c) to (K-1, c)
        dot_count = 0
        x_count = 0
        for r in range(K):
            if grid[r][c] == '.':
                dot_count += 1
            elif grid[r][c] == 'x':
                x_count += 1

        # Check the first window (starting row 0)
        if x_count == 0:
            min_ops = min(min_ops, dot_count)

        # Slide window to check segments starting from row 1 up to H-K
        # The starting row of the window is `start_r`.
        # When we move from starting row `start_r - 1` to `start_r`,
        # the cell at (start_r - 1, c) leaves and cell at (start_r + K - 1, c) enters.
        for start_r in range(1, H - K + 1):
            # Cell leaving: (start_r - 1, c)
            if grid[start_r - 1][c] == '.':
                dot_count -= 1
            elif grid[start_r - 1][c] == 'x':
                x_count -= 1

            # Cell entering: (start_r + K - 1, c)
            if grid[start_r + K - 1][c] == '.':
                dot_count += 1
            elif grid[start_r + K - 1][c] == 'x':
                x_count += 1

            # Check the current window starting at start_r
            if x_count == 0:
                min_ops = min(min_ops, dot_count)

# Output the result
if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)
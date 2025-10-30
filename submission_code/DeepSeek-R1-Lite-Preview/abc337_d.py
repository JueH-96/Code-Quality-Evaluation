import sys
import bisect

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    min_operations = float('inf')
    
    # Check horizontal windows
    for row in range(H):
        x_positions = [j for j, c in enumerate(grid[row]) if c == 'x']
        # Compute prefix sums of '.'s
        prefix_sum = [0] * (W + 1)
        for j in range(W):
            prefix_sum[j+1] = prefix_sum[j] + (1 if grid[row][j] == '.' else 0)
        # Slide the window
        for col in range(W - K + 1):
            # Check for 'x's in the window
            left = bisect.bisect_left(x_positions, col)
            right = bisect.bisect_right(x_positions, col + K - 1)
            if left < right:
                continue  # there is an 'x' in the window
            # No 'x's, count the number of '.'s
            num_dots = prefix_sum[col + K] - prefix_sum[col]
            if num_dots < min_operations:
                min_operations = num_dots
    
    # Check vertical windows
    for col in range(W):
        x_positions = [i for i in range(H) if grid[i][col] == 'x']
        # Compute prefix sums of '.'s
        prefix_sum = [0] * (H + 1)
        for i in range(H):
            prefix_sum[i+1] = prefix_sum[i] + (1 if grid[i][col] == '.' else 0)
        # Slide the window
        for row in range(H - K + 1):
            # Check for 'x's in the window
            left = bisect.bisect_left(x_positions, row)
            right = bisect.bisect_right(x_positions, row + K - 1)
            if left < right:
                continue  # there is an 'x' in the window
            # No 'x's, count the number of '.'s
            num_dots = prefix_sum[row + K] - prefix_sum[row]
            if num_dots < min_operations:
                min_operations = num_dots
    
    if min_operations == float('inf'):
        print(-1)
    else:
        print(min_operations)

if __name__ == '__main__':
    main()
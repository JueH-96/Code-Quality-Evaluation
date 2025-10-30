import sys

def solve():
    N = int(sys.stdin.readline())

    # Initialize the N x N grid with placeholder values (e.g., 0)
    grid = [[0 for _ in range(N)] for _ in range(N)]

    # Calculate the center coordinates for Takahashi
    center_r, center_c = N // 2, N // 2
    # Place Takahashi at the center
    grid[center_r][center_c] = 'T'

    # Initialize the first dragon part number
    current_val = 1
    
    # Define the initial boundaries for the spiral traversal
    top = 0
    bottom = N - 1
    left = 0
    right = N - 1

    # Total number of dragon parts to place (N*N - 1 cells)
    total_cells_to_fill = N * N - 1

    # Loop as long as there are numbers to place and valid boundaries for the spiral
    while current_val <= total_cells_to_fill and top <= bottom and left <= right:
        # 1. Move right (fill the top row of the current layer)
        for c in range(left, right + 1):
            # If all required numbers are placed, break early
            if current_val > total_cells_to_fill:
                break
            grid[top][c] = current_val
            current_val += 1
        top += 1 # Shrink the top boundary inwards
        if current_val > total_cells_to_fill: # Check again after completing the segment
            break

        # 2. Move down (fill the rightmost column of the current layer)
        for r in range(top, bottom + 1):
            if current_val > total_cells_to_fill:
                break
            grid[r][right] = current_val
            current_val += 1
        right -= 1 # Shrink the right boundary inwards
        if current_val > total_cells_to_fill:
            break

        # 3. Move left (fill the bottom row of the current layer)
        # Only proceed if there's still a valid bottom row to traverse
        if top <= bottom:
            for c in range(right, left - 1, -1): # Traverse backwards
                if current_val > total_cells_to_fill:
                    break
                grid[bottom][c] = current_val
                current_val += 1
            bottom -= 1 # Shrink the bottom boundary inwards
            if current_val > total_cells_to_fill:
                break

        # 4. Move up (fill the leftmost column of the current layer)
        # Only proceed if there's still a valid left column to traverse
        if left <= right:
            for r in range(bottom, top - 1, -1): # Traverse backwards
                if current_val > total_cells_to_fill:
                    break
                grid[r][left] = current_val
                current_val += 1
            left += 1 # Shrink the left boundary inwards
            if current_val > total_cells_to_fill:
                break
                
    # Print the final grid
    for r in range(N):
        # Convert each element to string before joining, in case of 'T'
        print(*(str(x) for x in grid[r]))

solve()
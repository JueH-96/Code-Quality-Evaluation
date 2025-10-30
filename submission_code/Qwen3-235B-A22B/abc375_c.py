import sys

def main():
    n = int(sys.stdin.readline())
    grid = [list(sys.stdin.readline().strip()) for _ in range(n)]
    
    for i in range(1, n // 2 + 1):
        start_row = i - 1
        end_row = n - i
        start_col = i - 1
        end_col = n - i
        
        ring = []
        # Collect top row left to right
        for col in range(start_col, end_col + 1):
            ring.append(grid[start_row][col])
        # Collect right column top to bottom (excluding top and bottom)
        for row in range(start_row + 1, end_row):
            ring.append(grid[row][end_col])
        # Collect bottom row right to left if applicable
        if start_row < end_row:
            for col in range(end_col, start_col - 1, -1):
                ring.append(grid[end_row][col])
        # Collect left column bottom to top (excluding top and bottom)
        for row in range(end_row - 1, start_row, -1):
            ring.append(grid[row][start_col])
        
        # Rotate the ring
        s = end_row - start_row + 1
        shift_amount = s - 1
        if len(ring) == 0:
            continue
        rotated_ring = ring[-shift_amount:] + ring[:-shift_amount]
        
        # Put back the rotated ring
        index = 0
        # Top row
        for col in range(start_col, end_col + 1):
            grid[start_row][col] = rotated_ring[index]
            index += 1
        # Right column
        for row in range(start_row + 1, end_row):
            grid[row][end_col] = rotated_ring[index]
            index += 1
        # Bottom row
        if start_row < end_row:
            for col in range(end_col, start_col - 1, -1):
                grid[end_row][col] = rotated_ring[index]
                index += 1
        # Left column
        for row in range(end_row - 1, start_row, -1):
            grid[row][start_col] = rotated_ring[index]
            index += 1
    
    # Print the result
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()
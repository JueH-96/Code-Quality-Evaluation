def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # Create an empty grid to hold strings (numbers or "T")
    grid = [["" for _ in range(N)] for _ in range(N)]
    # The center cell (Takahashi's cell) is at (N//2, N//2)
    center = (N // 2, N // 2)
    
    # We'll generate a spiral order of all cells.
    # In a spiral traversal of an odd×odd grid the very last cell is the center.
    order = []
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    while top <= bottom and left <= right:
        # traverse top row (left to right)
        for j in range(left, right + 1):
            order.append((top, j))
        top += 1
        
        # traverse right column (top to bottom)
        for i in range(top, bottom + 1):
            order.append((i, right))
        right -= 1
        
        if top <= bottom:
            # traverse bottom row (right to left)
            for j in range(right, left - 1, -1):
                order.append((bottom, j))
            bottom -= 1
        
        if left <= right:
            # traverse left column (bottom to top)
            for i in range(bottom, top - 1, -1):
                order.append((i, left))
            left += 1
    
    # Remove the center cell from the spiral order.
    # (Because in a spiral of an odd×odd grid the center cell is always last.)
    if order and order[-1] == center:
        order.pop()
    # Now, order has exactly N^2 - 1 coordinates for the dragon parts.
    
    # Assign dragon parts 1,2,...,N^2-1 along the path order.
    for idx, (i, j) in enumerate(order):
        grid[i][j] = str(idx + 1)
    
    # Place Takahashi in the center.
    ci, cj = center
    grid[ci][cj] = "T"
    
    # Print the grid.
    output = "
".join(" ".join(row) for row in grid)
    sys.stdout.write(output)

if __name__ == '__main__':
    main()
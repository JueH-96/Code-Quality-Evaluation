def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    grid_strings = data[2:]
    
    # Store the grid in a 2D list
    grid = [list(row) for row in grid_strings]
    
    # The word we are looking for
    target = "snuke"
    
    # Possible directions (dr, dc): 8 neighbors (vertical, horizontal, diagonal)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == target[0]:  # Found 's'
                for dr, dc in directions:
                    # Check if we can form "snuke" along this direction
                    found = True
                    positions = []
                    for i, ch in enumerate(target):
                        rr = r + dr*i
                        cc = c + dc*i
                        if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == ch:
                            positions.append((rr+1, cc+1))  # +1 for 1-based indexing
                        else:
                            found = False
                            break
                    if found:
                        # We found the unique set of 5 cells
                        for pos in positions:
                            print(pos[0], pos[1])
                        return  # Since there is a unique solution, we can stop here

# Do not forget to call main()
main()
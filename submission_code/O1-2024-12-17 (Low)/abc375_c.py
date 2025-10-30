def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    grid = data[1:]  # List of strings, each of length N
    
    # Convert grid to a list of lists for easy modification
    grid = [list(row) for row in grid]
    
    # We will rotate each "ring" by 90 degrees in place.
    # For 0-based indices, the i-th ring is bounded by [i, N-1-i] in both rows and columns.
    # The length of a side of this ring is M = (N - 2*i).
    # The perimeter is P = 4 * (M - 1).
    # A 90-degree clockwise rotation around this ring
    # is equivalent to rotating the ring's elements by (M - 1) positions in a clockwise listing.
    
    half = N // 2
    for i in range(half):
        top = i
        bottom = N - 1 - i
        left = i
        right = N - 1 - i
        
        M = right - left + 1  # dimension of this ring's square
        if M <= 1:
            continue
        P = 4 * (M - 1)      # perimeter of the ring
        offset = M - 1       # how many steps to rotate by 90 degrees
        
        # Collect the ring in clockwise order
        ring_coords = []
        
        # top row (left to right)
        for c in range(left, right + 1):
            ring_coords.append((top, c))
        # right column (top+1 to bottom-1)
        for r in range(top + 1, bottom):
            ring_coords.append((r, right))
        # bottom row (right to left)
        if bottom > top:  # avoid duplicate if it's a single row
            for c in range(right, left - 1, -1):
                ring_coords.append((bottom, c))
        # left column (bottom-1 to top+1)
        if left < right:  # avoid duplicate if it's a single column
            for r in range(bottom - 1, top, -1):
                ring_coords.append((r, left))
        
        # Extract the ring's colors
        ring_vals = [grid[r][c] for (r, c) in ring_coords]
        
        # Rotate the ring by offset positions (clockwise means we shift forward)
        # A simple way: new index = (old index + offset) mod P
        rotated_vals = [None]*P
        for idx in range(P):
            rotated_vals[(idx + offset) % P] = ring_vals[idx]
        
        # Place values back
        for idx, (r, c) in enumerate(ring_coords):
            grid[r][c] = rotated_vals[idx]
    
    # Print the resulting grid
    for row in grid:
        print("".join(row))

# Call main() as required
if __name__ == "__main__":
    main()
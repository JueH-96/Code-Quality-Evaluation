def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, Q = map(int, input_data[:3])
    queries = input_data[3:]
    
    # A convenient function to convert (r, c) to a 0-based index in a single list
    def idx(r, c):
        return (r - 1) * W + (c - 1)
    
    # Total number of cells (walls initially)
    n = H * W
    
    # Adjacency arrays: up[i], down[i], left[i], right[i] will give
    # the index of the cell in that direction, or -1 if none.
    up    = [-1] * n
    down  = [-1] * n
    left_ = [-1] * n
    right_ = [-1] * n
    
    # Initialize adjacency for each cell in row-major order
    # row r, col c => index i = r*W + c, r in [0..H-1], c in [0..W-1]
    # up/down differ by ±W, left/right differ by ±1
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if r > 0:
                up[i] = i - W
            else:
                up[i] = -1
            if r < H - 1:
                down[i] = i + W
            else:
                down[i] = -1
            
            if c > 0:
                left_[i] = i - 1
            else:
                left_[i] = -1
            if c < W - 1:
                right_[i] = i + 1
            else:
                right_[i] = -1
    
    # Track destroyed walls
    destroyed = [False] * n
    # Count of remaining walls
    rem_walls = n
    
    # A function to destroy a wall at index i (if it is not already destroyed).
    # This also "links" the neighbors so that future lookups skip this cell.
    def destroy(i):
        nonlocal rem_walls
        if destroyed[i]:
            return
        destroyed[i] = True
        rem_walls -= 1
        
        u = up[i]
        d = down[i]
        l = left_[i]
        r = right_[i]
        
        # Link up neighbor with down neighbor
        if u != -1:
            down[u] = d
        if d != -1:
            up[d] = u
        # Link left neighbor with right neighbor
        if l != -1:
            right_[l] = r
        if r != -1:
            left_[r] = l
    
    # Process queries
    p = 0
    for _ in range(Q):
        Rq = int(queries[p]); Cq = int(queries[p+1])
        p += 2
        
        cell = idx(Rq, Cq)
        # If the bombed cell still has a wall, just destroy it
        if not destroyed[cell]:
            destroy(cell)
        else:
            # Otherwise, destroy the first wall in each direction if any
            # Up
            up_idx = up[cell]
            if up_idx != -1:
                destroy(up_idx)
            # Down
            down_idx = down[cell]
            if down_idx != -1:
                destroy(down_idx)
            # Left
            left_idx = left_[cell]
            if left_idx != -1:
                destroy(left_idx)
            # Right
            right_idx = right_[cell]
            if right_idx != -1:
                destroy(right_idx)
    
    # Print the number of remaining walls
    print(rem_walls)

# Do not forget to call main()
if __name__ == "__main__":
    main()
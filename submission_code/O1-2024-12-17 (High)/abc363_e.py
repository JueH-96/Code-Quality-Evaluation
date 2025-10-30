def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Read H, W, Y
    H = int(input_data[0])
    W = int(input_data[1])
    Y = int(input_data[2])
    
    # Read the elevation matrix; collect squares up to height Y
    # because anything taller than Y will never sink within Y years.
    pos = 3
    squares = []
    for i in range(H):
        for j in range(W):
            h = int(input_data[pos])
            pos += 1
            if h <= Y:
                # Store (height, index)
                squares.append((h, i*W + j))
    
    # Sort squares in ascending order of height
    squares.sort(key=lambda x: x[0])
    
    # Total number of squares
    N = H * W
    
    # Union-Find (Disjoint Set) structures
    parent = list(range(N))       # parent[i] = parent of i
    size = [0]*N                  # size[i]   = size of the subtree if i is root
    boundary = [0]*N              # boundary[i] = 1 if the component touches the "sea boundary", else 0
    sea_size = [0]*N              # sea_size[i] = how many squares in this component are "sea"
    is_active = [0]*N             # is_active[i] = 1 if this square is "activated" (height <= current sea level)
    
    # Global sea_count: how many squares have sunk into the sea
    sea_count = 0
    
    # Iterative "find" with path compression
    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        # Path compression
        while parent[x] != r:
            px = parent[x]
            parent[x] = r
            x = px
        return r
    
    # Union by size; merges the sets of x and y
    def union(x, y):
        nonlocal sea_count
        rx = find(x)
        ry = find(y)
        if rx != ry:
            # Before merging, remember how many were already sea
            old_sea = sea_size[rx] + sea_size[ry]
            # Ensure rx is the larger root
            if size[rx] < size[ry]:
                rx, ry = ry, rx
            # Merge ry into rx
            parent[ry] = rx
            size[rx] += size[ry]
            # If either component touches boundary, merged one does too
            boundary[rx] |= boundary[ry]
            # If it touches boundary, entire component becomes sea
            if boundary[rx] == 1:
                sea_size[rx] = size[rx]
            else:
                sea_size[rx] = 0
            # Update global sea_count by the difference in sea sizes
            sea_count += sea_size[rx] - old_sea
    
    # Initialize a newly "activated" square in union-find
    def activate(x):
        nonlocal sea_count
        parent[x] = x
        size[x] = 1
        # Check if x is on the boundary of the island
        r, c = divmod(x, W)
        # Touches actual sea if it's on outermost row/col
        if r == 0 or r == H-1 or c == 0 or c == W-1:
            boundary[x] = 1
            sea_size[x] = 1
            sea_count += 1
        else:
            boundary[x] = 0
            sea_size[x] = 0
        is_active[x] = 1
    
    # Process squares in ascending order of height; answer queries i=1..Y
    p = 0
    len_squares = len(squares)
    results = []
    
    # A small helper to union with an already-active neighbor
    def try_union(x, y):
        if is_active[y] == 1:
            union(x, y)
    
    for level in range(1, Y+1):
        # Activate all squares whose height <= this sea level
        while p < len_squares and squares[p][0] <= level:
            x = squares[p][1]
            if is_active[x] == 0:
                activate(x)
                r, c = divmod(x, W)
                # Union with active neighbors
                if r > 0:      try_union(x, x - W)
                if r < H - 1:  try_union(x, x + W)
                if c > 0:      try_union(x, x - 1)
                if c < W - 1:  try_union(x, x + 1)
            p += 1
        
        # Number of squares still above sea = total minus how many have sunk
        results.append(str(N - sea_count))
    
    # Print the results
    print("
".join(results))

# Do not forget to call main()!
if __name__ == "__main__":
    main()
def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    cuboids = []
    idx = 1
    # We will collect cuboids as (X1, X2, Y1, Y2, Z1, Z2).
    for _ in range(N):
        x1 = int(input_data[idx]); y1 = int(input_data[idx+1]); z1 = int(input_data[idx+2])
        x2 = int(input_data[idx+3]); y2 = int(input_data[idx+4]); z2 = int(input_data[idx+5])
        idx += 6
        cuboids.append((x1, x2, y1, y2, z1, z2))
    
    # Since coordinates are all in [0..100] and cuboids don't overlap in volume,
    # we can afford a 3D grid up to 101^3 ~ 1,030,301 cells.
    # We'll store which cuboid occupies each cell (or 0 if none).
    SIZE = 101
    # Flattened 3D array: index = x*(SIZE*SIZE) + y*SIZE + z
    # Using a list for speed (rather than a nested list-of-lists in Python).
    grid = [0]* (SIZE*SIZE*SIZE)
    
    # Fill the grid with the IDs. The total volume is at most SIZE^3 = ~1e6,
    # which is feasible since the cuboids do not overlap.
    def index3d(x, y, z):
        return x*(SIZE*SIZE) + y*SIZE + z
    
    # Assign each cuboid's cells
    # i+1 will be the ID (1-based)
    for i, (x1, x2, y1, y2, z1, z2) in enumerate(cuboids, 1):
        for x in range(x1, x2):
            base_x = x*(SIZE*SIZE)
            for y in range(y1, y2):
                base_xy = base_x + y*SIZE
                for z in range(z1, z2):
                    grid[base_xy + z] = i  # Mark as occupied by cuboid i
    
    # We'll now check face-adjacencies in x, y, and z directions.
    # If grid[...] differs by exactly one coordinate, that means they share at least a 1x1 area on that face.
    adjacency_count = [0]* (N+1)
    
    # Check adjacency in X direction:
    for x in range(SIZE-1):  # up to 99
        base_x = x*(SIZE*SIZE)
        base_x_next = (x+1)*(SIZE*SIZE)
        for y in range(SIZE):  # 0..100
            base_xy = base_x + y*SIZE
            base_xy_next = base_x_next + y*SIZE
            for z in range(SIZE):  # 0..100
                c1 = grid[base_xy + z]
                c2 = grid[base_xy_next + z]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_count[c1] += 1
                    adjacency_count[c2] += 1
    
    # Check adjacency in Y direction:
    # We'll fix x, then for y..y+1, then z
    for x in range(SIZE):
        base_x = x*(SIZE*SIZE)
        for y in range(SIZE-1):  # up to 99
            base_xy = base_x + y*SIZE
            base_xy_next = base_x + (y+1)*SIZE
            for z in range(SIZE):
                c1 = grid[base_xy + z]
                c2 = grid[base_xy_next + z]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_count[c1] += 1
                    adjacency_count[c2] += 1
    
    # Check adjacency in Z direction:
    # We'll fix x, y, then compare z..z+1
    for x in range(SIZE):
        base_x = x*(SIZE*SIZE)
        for y in range(SIZE):
            base_xy = base_x + y*SIZE
            for z in range(SIZE-1):
                c1 = grid[base_xy + z]
                c2 = grid[base_xy + (z+1)]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    adjacency_count[c1] += 1
                    adjacency_count[c2] += 1
    
    # We counted each shared cell-boundary, but for a "face" to be shared,
    # we only need to know if there's at least one cell boundary in that plane.
    # In other words, if adjacency_count[c1] was incremented at least once for c2,
    # it means c1 and c2 share a face. Right now we are double-counting cell-boundaries
    # and even combining them across many squares. We need to reduce this to "did they ever share at least one boundary?".
    #
    # Let's re-work the approach: instead of adding +1 each time, we should record the pair (c1, c2) once only.
    # Then the final adjacency is the count of unique neighbors. We'll do sets of neighbors for each cuboid.
    
    # We'll redo the adjacency logic more carefully:
    neighbors = [set() for _ in range(N+1)]
    
    # X-direction adjacency
    for x in range(SIZE-1):
        base_x = x*(SIZE*SIZE)
        base_x_next = (x+1)*(SIZE*SIZE)
        for y in range(SIZE):
            base_xy = base_x + y*SIZE
            base_xy_next = base_x_next + y*SIZE
            for z in range(SIZE):
                c1 = grid[base_xy + z]
                c2 = grid[base_xy_next + z]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    neighbors[c1].add(c2)
                    neighbors[c2].add(c1)
    
    # Y-direction adjacency
    for x in range(SIZE):
        base_x = x*(SIZE*SIZE)
        for y in range(SIZE-1):
            base_xy = base_x + y*SIZE
            base_xy_next = base_x + (y+1)*SIZE
            for z in range(SIZE):
                c1 = grid[base_xy + z]
                c2 = grid[base_xy_next + z]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    neighbors[c1].add(c2)
                    neighbors[c2].add(c1)
    
    # Z-direction adjacency
    for x in range(SIZE):
        base_x = x*(SIZE*SIZE)
        for y in range(SIZE):
            base_xy = base_x + y*SIZE
            for z in range(SIZE-1):
                c1 = grid[base_xy + z]
                c2 = grid[base_xy + (z+1)]
                if c1 != 0 and c2 != 0 and c1 != c2:
                    neighbors[c1].add(c2)
                    neighbors[c2].add(c1)
    
    # Now the answer for cuboid i is the size of neighbors[i].
    # We'll print these in order i=1..N.
    out = []
    for i in range(1, N+1):
        out.append(str(len(neighbors[i])))
    print("
".join(out))

# Don't forget to call main().
if __name__ == "__main__":
    main()
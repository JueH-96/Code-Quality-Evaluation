# YOUR CODE HERE
import sys

def solve():
    """
    Solves the cuboid adjacency problem using a 3D grid.
    """
    # Fast I/O
    input = sys.stdin.readline

    # Read the number of cuboids
    try:
        N = int(input())
    except (ValueError, IndexError):
        # Handle empty input
        return
    
    if N == 0:
        return

    # Read all cuboid dimensions
    cuboids = [list(map(int, input().split())) for _ in range(N)]

    # The problem specifies coordinates are within [0, 100].
    # A cuboid from (x1, y1, z1) to (x2, y2, z2) occupies unit cubes
    # (i, j, k) for x1 <= i < x2, y1 <= j < y2, z1 <= k < z2.
    # The maximum index for i, j, k will be 99.
    # So a 100x100x100 grid is sufficient.
    DIM = 100
    D2 = DIM * DIM

    # We use a flattened 1D list for the 3D grid for better performance
    # in CPython compared to a list of lists of lists.
    # grid[x][y][z] is mapped to grid[x*D2 + y*DIM + z]
    grid = [0] * (DIM * D2)

    # Populate the grid with cuboid indices (1-based to distinguish from 0=empty)
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        for x in range(x1, x2):
            for y in range(y1, y2):
                base_idx = x * D2 + y * DIM
                for z in range(z1, z2):
                    grid[base_idx + z] = i + 1

    # adj will store sets of neighbors for each cuboid to handle duplicates
    adj = [set() for _ in range(N)]

    # Find adjacencies by checking neighboring cells in the grid.
    # We iterate through each axis direction to find adjacent pairs.
    
    # Check along X-axis
    for x in range(DIM - 1):
        for y in range(DIM):
            for z in range(DIM):
                idx1 = x * D2 + y * DIM + z
                idx2 = idx1 + D2
                id1, id2 = grid[idx1], grid[idx2]
                if id1 and id2 and id1 != id2:
                    # Convert 1-based grid IDs to 0-based list indices
                    c1_idx, c2_idx = id1 - 1, id2 - 1
                    adj[c1_idx].add(c2_idx)
                    adj[c2_idx].add(c1_idx)

    # Check along Y-axis
    for x in range(DIM):
        for y in range(DIM - 1):
            for z in range(DIM):
                idx1 = x * D2 + y * DIM + z
                idx2 = idx1 + DIM
                id1, id2 = grid[idx1], grid[idx2]
                if id1 and id2 and id1 != id2:
                    c1_idx, c2_idx = id1 - 1, id2 - 1
                    adj[c1_idx].add(c2_idx)
                    adj[c2_idx].add(c1_idx)

    # Check along Z-axis
    for x in range(DIM):
        for y in range(DIM):
            for z in range(DIM - 1):
                idx1 = x * D2 + y * DIM + z
                idx2 = idx1 + 1
                id1, id2 = grid[idx1], grid[idx2]
                if id1 and id2 and id1 != id2:
                    c1_idx, c2_idx = id1 - 1, id2 - 1
                    adj[c1_idx].add(c2_idx)
                    adj[c2_idx].add(c1_idx)

    # Print the number of unique neighbors for each cuboid
    for i in range(N):
        print(len(adj[i]))

solve()
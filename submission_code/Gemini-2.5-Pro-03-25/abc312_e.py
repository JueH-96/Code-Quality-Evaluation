# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read the number of cuboids
    N = int(sys.stdin.readline())
    
    # List to store coordinates of each cuboid
    cuboids_coords = []
    # Read coordinates for N cuboids
    for i in range(N):
        # Each line contains X1, Y1, Z1, X2, Y2, Z2 for the i-th cuboid
        line = list(map(int, sys.stdin.readline().split()))
        cuboids_coords.append(line)

    # Create a 3D grid representing the space.
    # The space is discretized into unit cubes [x, x+1) x [y, y+1) x [z, z+1).
    # The constraints state that coordinates are between 0 and 100.
    # A cuboid [X1, X2) x [Y1, Y2) x [Z1, Z2) covers unit cubes with indices (x, y, z)
    # such that X1 <= x < X2, Y1 <= y < Y2, Z1 <= z < Z2.
    # Since X2, Y2, Z2 can be up to 100, the maximum index needed is 99.
    # Thus, the grid dimensions are 100x100x100.
    # grid[x][y][z] stores the ID (1 to N) of the cuboid occupying the unit cube at index (x, y, z).
    # 0 indicates the unit cube is empty space.
    grid = [[[0 for _ in range(100)] for _ in range(100)] for _ in range(100)] 

    # Fill the grid with cuboid IDs
    for i in range(N):
        # Use 1-based index for cuboid ID, as the problem indices start from 1
        # and output should correspond to these original indices.
        idx = i + 1 
        x1, y1, z1, x2, y2, z2 = cuboids_coords[i]
        
        # Iterate through the integer coordinates (x, y, z) defining the unit cubes
        # covered by the current cuboid. The ranges are [x1, x2), [y1, y2), [z1, z2).
        # The constraints 0 <= X1 < X2 <= 100 etc. ensure x, y, z indices will be within [0, 99].
        for x in range(x1, x2):
            for y in range(y1, y2):
                for z in range(z1, z2):
                    # Assign the cuboid ID to the corresponding grid cell.
                    # The problem guarantees that cuboids do not overlap by volume,
                    # so each grid cell corresponding to a unit cube inside any cuboid
                    # will be assigned an ID exactly once.
                    grid[x][y][z] = idx

    # Initialize an adjacency list. adj[i] will store a set of unique cuboid IDs 
    # that share a face with cuboid i.
    # Using sets automatically handles uniqueness of neighbors.
    # The list size is N+1 to accommodate 1-based cuboid IDs (index 0 is unused).
    adj = [set() for _ in range(N + 1)] 

    # Iterate through all grid cells to find pairs of adjacent cuboids.
    # Two cuboids share a face if there exist two adjacent unit cubes,
    # one belonging to each cuboid.
    # Checking neighbors only in positive directions (+x, +y, +z) is sufficient.
    # Every boundary between two adjacent cells (A, B) will be checked exactly once:
    # when visiting the cell with the smaller coordinate(s) (e.g., A).
    # The adjacency is then recorded for both A and B.
    for x in range(100):
        for y in range(100):
            for z in range(100):
                # Get the ID of the cuboid occupying the current cell (x, y, z)
                current_cuboid_id = grid[x][y][z]
                
                # If the current cell is empty (ID=0), skip it
                if current_cuboid_id == 0: 
                    continue

                # Check neighbor cell in the positive x direction: (x+1, y, z)
                if x + 1 < 100: # Check if x+1 is within grid bounds [0, 99]
                    neighbor_cuboid_id = grid[x + 1][y][z]
                    # Check if the neighbor cell is occupied (ID > 0) AND by a *different* cuboid
                    if neighbor_cuboid_id != 0 and neighbor_cuboid_id != current_cuboid_id:
                        # If adjacent and different, record the adjacency for both cuboids
                        adj[current_cuboid_id].add(neighbor_cuboid_id)
                        adj[neighbor_cuboid_id].add(current_cuboid_id) # Adjacency is symmetric

                # Check neighbor cell in the positive y direction: (x, y+1, z)
                if y + 1 < 100: # Check if y+1 is within grid bounds [0, 99]
                    neighbor_cuboid_id = grid[x][y + 1][z]
                    if neighbor_cuboid_id != 0 and neighbor_cuboid_id != current_cuboid_id:
                        adj[current_cuboid_id].add(neighbor_cuboid_id)
                        adj[neighbor_cuboid_id].add(current_cuboid_id)
                
                # Check neighbor cell in the positive z direction: (x, y, z+1)
                if z + 1 < 100: # Check if z+1 is within grid bounds [0, 99]
                    neighbor_cuboid_id = grid[x][y][z + 1]
                    if neighbor_cuboid_id != 0 and neighbor_cuboid_id != current_cuboid_id:
                        adj[current_cuboid_id].add(neighbor_cuboid_id)
                        adj[neighbor_cuboid_id].add(current_cuboid_id)

    # Prepare the final results for printing
    results = []
    # Iterate through cuboid IDs from 1 to N
    for i in range(1, N + 1):
       # The number of neighbors for cuboid i is the size of its adjacency set
       results.append(str(len(adj[i])))

    # Print the counts, each on a new line, as required by the output format
    print("
".join(results))

# Execute the main function to solve the problem
solve()
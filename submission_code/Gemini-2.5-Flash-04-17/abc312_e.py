import sys

def solve():
    N = int(sys.stdin.readline())
    cuboids = []
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, sys.stdin.readline().split())
        cuboids.append((x1, y1, z1, x2, y2, z2))

    # Maps to store which cuboid face covers an elementary square on a plane
    # Key: (plane_coord, coord1, coord2) -> cuboid_idx
    # plane_coord is the fixed coordinate of the plane (e.g., x for an x-plane)
    # coord1, coord2 are the coordinates spanning the plane (e.g., y, z for an x-plane)
    face_map_x_right = {} # (x, y, z) -> cuboid_idx
    face_map_x_left = {}  # (x, y, z) -> cuboid_idx
    face_map_y_top = {}   # (y, x, z) -> cuboid_idx
    face_map_y_bottom = {}# (y, x, z) -> cuboid_idx
    face_map_z_front = {} # (z, x, y) -> cuboid_idx
    face_map_z_back = {}  # (z, x, y) -> cuboid_idx

    # Populate the maps
    for idx, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        # Right face at x2
        # Elementary squares on this face are [x2, x2] x [y, y+1] x [z, z+1]
        # where y ranges from y1 to y2-1 and z ranges from z1 to z2-1.
        for y in range(y1, y2):
            for z in range(z1, z2):
                face_map_x_right[(x2, y, z)] = idx
        # Left face at x1
        # Elementary squares on this face are [x1, x1] x [y, y+1] x [z, z+1]
        # where y ranges from y1 to y2-1 and z ranges from z1 to z2-1.
        for y in range(y1, y2):
            for z in range(z1, z2):
                face_map_x_left[(x1, y, z)] = idx
        # Top face at y2
        # Elementary squares on this face are [x, x+1] x [y2, y2] x [z, z+1]
        # where x ranges from x1 to x2-1 and z ranges from z1 to z2-1.
        for x in range(x1, x2):
            for z in range(z1, z2):
                face_map_y_top[(y2, x, z)] = idx
        # Bottom face at y1
        # Elementary squares on this face are [x, x+1] x [y1, y1] x [z, z+1]
        # where x ranges from x1 to x2-1 and z ranges from z1 to z2-1.
        for x in range(x1, x2):
            for z in range(z1, z2):
                face_map_y_bottom[(y1, x, z)] = idx
        # Front face at z2
        # Elementary squares on this face are [x, x+1] x [y, y+1] x [z2, z2]
        # where x ranges from x1 to x2-1 and y ranges from y1 to y2-1.
        for x in range(x1, x2):
            for y in range(y1, y2):
                face_map_z_front[(z2, x, y)] = idx
        # Back face at z1
        # Elementary squares on this face are [x, x+1] x [y, y+1] x [z1, z1]
        # where x ranges from x1 to x2-1 and y ranges from y1 to y2-1.
        for x in range(x1, x2):
            for y in range(y1, y2):
                face_map_z_back[(z1, x, y)] = idx

    neighbor_pairs = set()

    # Check X-planes (x=const)
    # Iterate through potential planes x=0 to x=100
    # Iterate through elementary squares [x, x] x [y, y+1] x [z, z+1]
    # The bottom-left corner of the square on the YZ plane is (y, z).
    for x in range(0, 101): # Plane coordinate (0 to 100)
        for y in range(0, 100): # First coordinate on plane (0 to 99)
            for z in range(0, 100): # Second coordinate on plane (0 to 99)
                idx1 = face_map_x_right.get((x, y, z))
                idx2 = face_map_x_left.get((x, y, z))
                if idx1 is not None and idx2 is not None:
                    neighbor_pairs.add(tuple(sorted((idx1, idx2))))

    # Check Y-planes (y=const)
    # Iterate through potential planes y=0 to y=100
    # Iterate through elementary squares [x, x+1] x [y, y] x [z, z+1]
    # The bottom-left corner of the square on the XZ plane is (x, z).
    for y in range(0, 101): # Plane coordinate (0 to 100)
        for x in range(0, 100): # First coordinate on plane (0 to 99)
            for z in range(0, 100): # Second coordinate on plane (0 to 99)
                 idx1 = face_map_y_top.get((y, x, z))
                 idx2 = face_map_y_bottom.get((y, x, z))
                 if idx1 is not None and idx2 is not None:
                     neighbor_pairs.add(tuple(sorted((idx1, idx2))))

    # Check Z-planes (z=const)
    # Iterate through potential planes z=0 to z=100
    # Iterate through elementary squares [x, x+1] x [y, y+1] x [z, z]
    # The bottom-left corner of the square on the XY plane is (x, y).
    for z in range(0, 101): # Plane coordinate (0 to 100)
        for x in range(0, 100): # First coordinate on plane (0 to 99)
            for y in range(0, 100): # Second coordinate on plane (0 to 99)
                 idx1 = face_map_z_front.get((z, x, y))
                 idx2 = face_map_z_back.get((z, x, y))
                 if idx1 is not None and idx2 is not None:
                     neighbor_pairs.add(tuple(sorted((idx1, idx2))))

    neighbor_counts = [0] * N
    for i, j in neighbor_pairs:
        neighbor_counts[i] += 1
        neighbor_counts[j] += 1

    for count in neighbor_counts:
        print(count)

solve()
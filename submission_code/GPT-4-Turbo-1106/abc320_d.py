import sys

def find_coordinates(N, M, info):
    # Initialize coordinates with None, except for person 1 at the origin
    coordinates = {i: None for i in range(1, N + 1)}
    coordinates[1] = (0, 0)

    # Process the information
    for A, B, X, Y in info:
        if coordinates[A] is not None:
            # Calculate the coordinates of B based on A's location and the offsets
            new_coords = (coordinates[A][0] + X, coordinates[A][1] + Y)
            if coordinates[B] is None:
                coordinates[B] = new_coords
            elif coordinates[B] != new_coords:
                # If there is a conflict in the information, it's undecidable
                return ["undecidable"] * N
        else:
            # If A's coordinates are not known, we cannot determine B's coordinates yet
            pass

    # Check if all coordinates are determined
    for i in range(1, N + 1):
        if coordinates[i] is None:
            coordinates[i] = "undecidable"

    return coordinates

# Read input
N, M = map(int, sys.stdin.readline().split())
info = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

# Find coordinates
coordinates = find_coordinates(N, M, info)

# Output results
for i in range(1, N + 1):
    if coordinates[i] == "undecidable":
        print(coordinates[i])
    else:
        print(f"{coordinates[i][0]} {coordinates[i][1]}")
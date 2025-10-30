import sys
from collections import deque, defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list: adj[u] will store a list of (v, dx, dy) tuples,
    # meaning if u is at (x_u, y_u), then v is at (x_u + dx, y_u + dy).
    adj = defaultdict(list)
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        # Information: B is X units right, Y units up from A
        adj[A].append((B, X, Y))
        # Implied inverse information: A is -X units right, -Y units up from B
        adj[B].append((A, -X, -Y))

    # coords[i] will store the (x, y) coordinates for person i.
    # Initialize with None to indicate that coordinates are not yet determined.
    # Using 1-based indexing for people, so coords array size N+1.
    coords = [None] * (N + 1)

    # Person 1 is at the origin (0, 0)
    coords[1] = (0, 0)

    # Queue for BFS: stores (person_id, x_coord, y_coord)
    # Start BFS from person 1, whose position is known.
    q = deque()
    q.append((1, 0, 0))

    while q:
        u, x_u, y_u = q.popleft()

        # Explore neighbors of u
        for v, dx, dy in adj[u]:
            # Calculate v's coordinates based on u's coordinates and relative displacement
            x_v = x_u + dx
            y_v = y_u + dy

            # If v's coordinates haven't been determined yet (first time visiting v via BFS from a determined path)
            if coords[v] is None:
                coords[v] = (x_v, y_v)
                q.append((v, x_v, y_v))
            # Else, v has already been visited and its coordinates determined.
            # According to the problem statement, "The given information is consistent,"
            # so coords[v] must be equal to (x_v, y_v). No explicit consistency check or conflict resolution is needed.

    # Prepare output strings
    results = []
    for i in range(1, N + 1):
        if coords[i] is None:
            # If coords[i] is still None, it means person i was not reachable from person 1
            # through the given relative position information.
            # Therefore, their coordinates cannot be uniquely determined relative to the origin.
            results.append("undecidable")
        else:
            # Coordinates are uniquely determined.
            results.append(f"{coords[i][0]} {coords[i][1]}")
    
    # Print all results, each on a new line.
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program.
solve()
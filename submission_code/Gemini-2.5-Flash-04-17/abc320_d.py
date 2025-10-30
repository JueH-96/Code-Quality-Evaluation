# YOUR CODE HERE
import collections
import sys

# Function to solve the problem
def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Build adjacency list representation of the graph
    # adj[u] will store a list of (v, dx, dy) tuples,
    # meaning person v is located at (coords[u][0] + dx, coords[u][1] + dy)
    # if coords[u] is known.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        # Constraint: Position of B relative to A is (X, Y)
        # If A is at (px_A, py_A), B is at (px_A + X, py_A + Y).
        # Edge from A to B with displacement (X, Y).
        adj[A].append((B, X, Y))
        # Conversely, if B is at (px_B, py_B), A is at (px_B - X, py_B - Y).
        # Edge from B to A with displacement (-X, -Y).
        adj[B].append((A, -X, -Y))

    # List to store the determined coordinates for each person.
    # coords[i] will store a tuple (px_i, py_i) or None if undetermined.
    # Use size N+1 for 1-based indexing.
    coords = [None] * (N + 1)

    # Person 1 is fixed at the origin (0, 0)
    coords[1] = (0, 0)

    # Queue for Breadth-First Search (BFS).
    # Start with person 1 whose coordinates are known.
    q = collections.deque([1])

    # Perform BFS traversal starting from person 1
    # This will visit all people whose coordinates can be determined relative to person 1.
    while q:
        # Get the next person from the queue whose coordinates are determined
        u = q.popleft()

        # Get the coordinates of person u
        px_u, py_u = coords[u]

        # Explore neighbors of person u
        for v, dx, dy in adj[u]:
            # If coordinates of person v are not yet determined
            if coords[v] is None:
                # Determine coordinates of v based on u's coordinates and the displacement
                coords[v] = (px_u + dx, py_u + dy)
                # Add v to the queue to explore its neighbors next
                q.append(v)
            # Else: coords[v] is already determined.
            # Since the problem guarantees consistency, we don't need to perform any checks.
            # If inconsistency was possible, we would check if the newly calculated coordinates
            # match the existing ones for v.

    # Output the results for each person from 1 to N
    for i in range(1, N + 1):
        if coords[i] is None:
            # Coordinates could not be determined (person is not in the same component as person 1)
            sys.stdout.write("undecidable
")
        else:
            # Coordinates are determined
            sys.stdout.write(f"{coords[i][0]} {coords[i][1]}
")

# Execute the solve function
solve()
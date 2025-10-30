import sys
import collections

# Increase recursion depth if needed, though BFS is generally preferred for this type of problem
# and does not rely on recursion depth. Python's default recursion depth is usually sufficient.
# sys.setrecursionlimit(4 * 10**5 + 10) 

def solve():
    # Read number of people N and number of information pieces M
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize adjacency list representation of the graph.
    # adj[u] will store a list of tuples (v, dx, dy), representing a directed edge
    # from person u to person v, indicating that v's position relative to u is (dx, dy).
    adj = [[] for _ in range(N + 1)] # Use index 1 to N
    
    # Process each piece of information
    for _ in range(M):
        # Read the relation: A_i, B_i, X_i, Y_i
        # This means from person A_i's perspective, person B_i is X_i units in positive x
        # and Y_i units in positive y direction.
        A, B, X, Y = map(int, sys.stdin.readline().split())
        
        # This translates to: position(B) = position(A) + (X, Y)
        # We can model this with directed edges:
        # Add edge A -> B with weight (X, Y)
        adj[A].append((B, X, Y))
        
        # This relation also implies: position(A) = position(B) - (X, Y) = position(B) + (-X, -Y)
        # Add edge B -> A with weight (-X, -Y) to allow traversal in the reverse direction
        adj[B].append((A, -X, -Y))

    # Array to store the coordinates (x, y) for each person.
    # Initialize all coordinates to None, indicating they are undetermined.
    coords = [None] * (N + 1) # Use index 1 to N
    
    # Base case: Person 1 is at the origin (0, 0).
    # Set their coordinates and mark them as determined.
    coords[1] = (0, 0)
    
    # Initialize a queue for Breadth-First Search (BFS).
    # Start the BFS from person 1, whose position is known.
    q = collections.deque([1])
    
    # Perform BFS to determine the coordinates of all people reachable from person 1.
    while q:
        # Get the next person `u` from the front of the queue.
        u = q.popleft()
        
        # Retrieve the coordinates of person `u`. These are guaranteed to be determined
        # because `u` was added to the queue only after its coordinates were set.
        ux, uy = coords[u] 
        
        # Explore all neighbors `v` of person `u`. A neighbor `v` is a person whose
        # relative position to `u` is known (or vice versa).
        for v, dx, dy in adj[u]:
            # Check if person `v`'s coordinates have already been determined.
            if coords[v] is None:
                # If coordinates are not determined (`v` has not been visited yet):
                # Calculate `v`'s coordinates based on `u`'s coordinates `(ux, uy)`
                # and the relative displacement `(dx, dy)`.
                coords[v] = (ux + dx, uy + dy)
                
                # Add person `v` to the queue to explore its neighbors later.
                q.append(v)
            # If coords[v] is not None, it means `v` has already been visited and its coordinates determined.
            # The problem statement guarantees consistency, so we don't need to check for conflicting information
            # or update coordinates. The BFS structure ensures we find the coordinates via a shortest path
            # in terms of number of edges, but any path from person 1 yields the same coordinates due to consistency.

    # After BFS, prepare the output strings for each person.
    output_lines = []
    for i in range(1, N + 1):
        # Check if coordinates for person `i` were determined during BFS.
        if coords[i] is None:
            # If coordinates remain None, it means person `i` is not reachable from person 1
            # through the given relative position information. Thus, their position is undecidable.
            output_lines.append("undecidable")
        else:
            # If coordinates were determined, format them as a string "x y".
            output_lines.append(f"{coords[i][0]} {coords[i][1]}")
    
    # Print all output lines together, separated by newlines, for efficiency.
    print('
'.join(output_lines))

# Execute the solver function to run the program.
solve()
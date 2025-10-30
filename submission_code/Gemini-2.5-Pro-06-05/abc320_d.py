# YOUR CODE HERE
import sys
from collections import deque

# For competitive programming, it's a good practice to increase the recursion
# limit, especially for graph problems that could be solved with a deep recursive
# DFS. While this solution uses iterative BFS, this line is a good habit.
sys.setrecursionlimit(2 * 10**6)

def main():
    """
    Main function to orchestrate reading input, solving the problem, and printing output.
    """
    # Use a faster I/O method for large inputs
    input = sys.stdin.readline
    
    # Read the number of people (N) and the number of relations (M)
    try:
        line = input()
        if not line.strip():
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Gracefully exit on empty or malformed input
        return

    # Create an adjacency list to represent the graph of people.
    # adj[u] will store a list of tuples: (v, dx, dy),
    # meaning person v's position relative to person u is (dx, dy).
    adj = [[] for _ in range(N)]

    for _ in range(M):
        # Read the relation: A, B, X, Y
        A, B, X, Y = map(int, input().split())
        # Convert 1-based person numbers to 0-based list indices
        u, v = A - 1, B - 1
        
        # The relation "from A's perspective, B is (X, Y) away" means:
        # pos(B) = pos(A) + (X, Y)  =>  pos(v) = pos(u) + (X, Y)
        # We store this as a directed edge u -> v with a "weight" of (X, Y).
        adj[u].append((v, X, Y))
        
        # From this, we can also infer the reverse relationship:
        # pos(A) = pos(B) - (X, Y)  =>  pos(u) = pos(v) + (-X, -Y)
        # We store this as a directed edge v -> u with weight (-X, -Y).
        # Adding this reverse edge makes the graph effectively undirected for traversal purposes.
        adj[v].append((u, -X, -Y))

    # `coords` will store the absolute coordinates (x, y) for each person.
    # Initialize with `None` to signify that coordinates are initially unknown.
    # `None` also conveniently serves as our "not visited" marker for the BFS.
    coords = [None] * N
    
    # The problem statement gives a crucial piece of information:
    # Person 1 (index 0) is at the origin (0, 0). This is our anchor point.
    coords[0] = (0, 0)
    
    # We will use a queue for a Breadth-First Search (BFS). BFS is a great
    # choice here because it naturally finds all reachable nodes from a source
    # and we can propagate the coordinate information layer by layer.
    q = deque([0])
    
    while q:
        # Get the next person from the queue to process
        u = q.popleft()
        x_u, y_u = coords[u]
        
        # For each person `v` connected to `u`
        for v, dx, dy in adj[u]:
            # If we haven't determined v's coordinates yet...
            if coords[v] is None:
                # ...we can now calculate them based on u's known absolute
                # position and the relative offset (dx, dy).
                coords[v] = (x_u + dx, y_u + dy)
                # Add `v` to the queue so we can process its neighbors later.
                q.append(v)
    
    # After the BFS completes, `coords` is populated for everyone
    # in the same connected component as person 1.
    # Anyone not in that component will still have `None` as their coordinate value.
    # These are the "undecidable" cases.
    for coord in coords:
        if coord is None:
            print("undecidable")
        else:
            x_i, y_i = coord
            print(f"{x_i} {y_i}")

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys
import collections

# It's recommended to increase recursion limit for deep graphs,
# although we will use BFS (iteration) here to be safe.
# For competitive programming, this is a good habit.
sys.setrecursionlimit(2 * 10**5 + 5)

def solve():
    """
    This function encapsulates the entire solution.
    It reads from stdin, solves the problem, and prints to stdout.
    """
    
    # Fast I/O
    # Use try-except for compatibility with environments where stdin.readline is not available.
    try:
        from sys import stdin
        input = stdin.readline
    except ImportError:
        pass

    # Read problem input
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
    except (IOError, ValueError):
        return

    # Adjacency list to represent the tree
    adj = collections.defaultdict(list)
    # Array to store the degree of each vertex
    degrees = [0] * (N + 1)

    for _ in range(N - 1):
        try:
            u, v = map(int, input().split())
        except (IOError, ValueError):
            # Handle empty lines or malformed input at the end of the file
            continue
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # --- The Core Logic ---
    # We can determine the original role of each vertex (center or leaf) by a propagation algorithm.
    # A vertex in the original graph was either a "center" of a star or a "leaf" of a star.
    # Let's define constants for these types.
    UNKNOWN = 0
    CENTER = 1
    ORIGINAL_LEAF = 2 # The leaves of the initial stars

    # `types` array will store the determined role of each vertex.
    types = [UNKNOWN] * (N + 1)
    # A queue for our Breadth-First Search (BFS) propagation.
    q = collections.deque()

    # --- Seeding Phase ---
    # We need to find some vertices whose roles are certain, to start the propagation.
    # 1. Any vertex with degree > 2 in the final tree T must have been a center.
    # 2. The vertices adjacent to the leaves of the final tree T must also have been centers.

    # Seed based on degree > 2
    for i in range(1, N + 1):
        if degrees[i] > 2:
            if types[i] == UNKNOWN:
                types[i] = CENTER
                q.append(i)

    # Seed based on neighbors of tree leaves (vertices with degree 1 in T)
    for i in range(1, N + 1):
        if degrees[i] == 1:
            # A degree-1 vertex in T must have been an ORIGINAL_LEAF that was not used for connections.
            types[i] = ORIGINAL_LEAF
            # Its only neighbor must be the center of the star it belonged to.
            neighbor = adj[i][0]
            if types[neighbor] == UNKNOWN:
                types[neighbor] = CENTER
                q.append(neighbor)
    
    # --- Propagation Phase (BFS) ---
    # Propagate the type information through the tree.
    while q:
        u = q.popleft()

        if types[u] == CENTER:
            # If u is a center, all its neighbors in T must have been its original leaves.
            for v in adj[u]:
                if types[v] == UNKNOWN:
                    types[v] = ORIGINAL_LEAF
                    q.append(v)
        
        elif types[u] == ORIGINAL_LEAF:
            # If u was an original leaf, it was connected to one center and (possibly) one other original leaf.
            # We can use this to deduce the types of its neighbors.
            center_neighbor = None
            unknown_neighbors = []
            
            for v in adj[u]:
                if types[v] == CENTER:
                    center_neighbor = v
                elif types[v] == UNKNOWN:
                    unknown_neighbors.append(v)
            
            if center_neighbor is not None:
                # The center is known, so all other unknown neighbors must be ORIGINAL_LEAFs
                # (they are connection partners).
                for v in unknown_neighbors:
                    types[v] = ORIGINAL_LEAF
                    q.append(v)
            elif len(unknown_neighbors) == 1:
                # The center is not known, but only one neighbor is of unknown type.
                # This neighbor must be the center.
                v = unknown_neighbors[0]
                types[v] = CENTER
                q.append(v)

    # --- Collection Phase ---
    # Find all vertices identified as centers and collect their levels.
    # The level of a star is the degree of its center in the final tree.
    levels = []
    for i in range(1, N + 1):
        if types[i] == CENTER:
            levels.append(degrees[i])
    
    # Sort the levels in ascending order as required by the problem.
    levels.sort()
    
    # Print the result.
    print(*levels)


# Execute the solution
solve()
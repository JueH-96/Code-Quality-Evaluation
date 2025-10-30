import sys
from collections import deque

# It's good practice to increase the recursion limit for deep trees,
# although this solution uses BFS and doesn't rely on deep recursion.
sys.setrecursionlimit(2 * 10**5 + 10)

def main():
    """
    Reads the tree structure, finds the largest possible alkane subgraph,
    and prints its size or -1 if none exists.
    """
    try:
        # Read N from the first line of stdin
        line = sys.stdin.readline()
        if not line:
            # Handle empty input case
            print(-1)
            return
        N = int(line)
    except (IOError, ValueError):
        # Handle cases where N is not a valid integer
        print(-1)
        return

    # Adjacency list for the original tree T
    adj = [[] for _ in range(N + 1)]
    # Array to store degrees of each vertex
    degrees = [0] * (N + 1)

    # Read N-1 edges from stdin
    for _ in range(N - 1):
        try:
            A, B = map(int, sys.stdin.readline().split())
            adj[A].append(B)
            adj[B].append(A)
            degrees[A] += 1
            degrees[B] += 1
        except (IOError, ValueError):
            # This part is for robustness against malformed input,
            # though problem constraints usually guarantee valid input.
            pass

    # A vertex in an alkane subgraph can have degree 4 only if its degree
    # in the original tree is also at least 4.
    # We call these "potential core" vertices.
    potential_cores = {i for i in range(1, N + 1) if degrees[i] >= 4}

    # An alkane must have at least one vertex of degree 4.
    # If there are no potential core vertices, no alkane is possible.
    if not potential_cores:
        print(-1)
        return

    # Construct the graph G' whose vertices are the potential cores and an edge
    # exists if they are adjacent in the original tree T.
    adj_cores = {u: [] for u in potential_cores}
    for u in potential_cores:
        for v in adj[u]:
            if v in potential_cores:
                adj_cores[u].append(v)
    
    max_comp_size = 0
    visited = set()

    # The set of degree-4 vertices in any alkane must form a connected component.
    # To maximize the alkane size, we need to find the largest possible set of
    # core vertices that can form a connected component. This corresponds to
    # finding the largest connected component in G'.
    for core_node in potential_cores:
        if core_node not in visited:
            current_comp_size = 0
            q = deque([core_node])
            visited.add(core_node)
            
            while q:
                u = q.popleft()
                current_comp_size += 1
                
                # We can safely access adj_cores[u] because all potential cores
                # were added as keys during initialization.
                for v in adj_cores[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            
            max_comp_size = max(max_comp_size, current_comp_size)

    # For a set of `k` core vertices forming a tree, the total number of vertices
    # in the resulting alkane is `k` (cores) plus the number of leaves.
    # Number of leaves = sum(4 - deg_core(v) for v in cores).
    # sum(deg_core(v)) = 2 * (k-1) for a tree with k vertices.
    # Total vertices = k + 4*k - 2*(k-1) = 3*k + 2.
    result = 3 * max_comp_size + 2
    print(result)

if __name__ == "__main__":
    main()
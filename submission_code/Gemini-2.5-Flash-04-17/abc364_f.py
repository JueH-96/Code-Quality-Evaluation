import sys

# Increase recursion depth for DSU path compression
# Maximum N+Q is 4 * 10^5. A slightly larger limit is safe.
sys.setrecursionlimit(400000 + 100)

# Disjoint Set Union (DSU) with path compression and union by size,
# augmented with max_idx in component for vertices 1..N
class DSU:
    def __init__(self, n, N_vertices):
        # parent[i]: parent of node i
        self.parent = list(range(n + 1))
        # size[i]: size of the component rooted at i (used for union by size)
        self.size = [1] * (n + 1)
        # max_idx[i]: maximum vertex index k in [1, N_vertices] that is in the component rooted at i
        self.max_idx = list(range(n + 1))
        # Initialize max_idx for vertices N_vertices + 1 to N + Q to 0 (or any value < 1)
        # because they initially contain no vertices from [1, N_vertices].
        # This ensures max(max_idx_root_i, max_idx_root_j) logic works when merging with a component from 1..N.
        for i in range(N_vertices + 1, n + 1):
            self.max_idx[i] = 0

    def find(self, i):
        # Base case: i is the root of its component
        if self.parent[i] == i:
            return i
        # Path compression: Set parent[i] directly to the root
        # We don't need to update max_idx here, it's only stored and updated at the root.
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    # Performs union of the sets containing i and j.
    # Returns True if a union occurred (i.e., i and j were in different sets), False otherwise.
    def unite(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        # If i and j are already in the same set, do nothing
        if root_i == root_j:
            return False # No union occurred

        # Union by size: Attach the smaller tree to the root of the larger tree
        if self.size[root_i] < self.size[root_j]:
            root_i, root_j = root_j, root_i

        # Merge root_j's component into root_i's component
        self.parent[root_j] = root_i
        self.size[root_i] += self.size[root_j]

        # Update max_idx of the new root: it's the maximum of the max_idx of the old roots
        # This correctly tracks the maximum vertex index from [1, N_vertices] in the merged component
        self.max_idx[root_i] = max(self.max_idx[root_i], self.max_idx[root_j])

        return True # Union successful

def solve():
    # Read N and Q
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    Q = int(line1[1])

    # Read queries
    queries = []
    for i in range(Q):
        line = sys.stdin.readline().split()
        L = int(line[0])
        R = int(line[1])
        C = int(line[2])
        # Store (cost, L, R, 1-based query_index)
        queries.append((C, L, R, i + 1))

    # Sort queries by cost in ascending order
    queries.sort()

    # Initialize DSU for N + Q vertices, numbered 1 to N+Q.
    # Pass N_vertices = N to DSU constructor to handle max_idx correctly.
    dsu = DSU(N + Q, N)

    mst_cost = 0 # Total cost of the Minimum Spanning Tree. Use Python's arbitrary precision integer.
    mst_edges = 0 # Number of edges added to the MST

    # Process queries in increasing order of cost (Kruskal's algorithm idea)
    for cost, L, R, q_idx in queries:
        # Vertex N + q_idx is vertex U in the graph (1-based indexing)
        U = N + q_idx

        # Iterate through vertices in the range [L, R] connected to U.
        # Use the optimized jump based on max_idx in components.
        j = L # Start checking from the left bound of the range

        # Continue as long as we are within the range [L, R]
        while j <= R:
            # Find the root of vertex j. This might involve path compression.
            root_j = dsu.find(j)
            # Find the root of vertex U.
            root_U = dsu.find(U)

            # If vertex j and vertex U are not already in the same component
            if root_j != root_U:
                # Attempt to unite their components. This corresponds to considering adding an edge (U, j) with cost C.
                # The unite operation only succeeds if they were genuinely in different sets.
                if dsu.unite(U, j):
                    # If union happened, we added a necessary edge to the MST at this cost.
                    mst_cost += cost
                    mst_edges += 1

            # Determine the next vertex to check in the range [L, R].
            # All vertices in the component containing j, up to max_idx[find(j)], are now effectively processed
            # relative to connecting to U's component at this cost level.
            # The next vertex we need to consider is the one immediately after this block of indices from 1..N.
            # Use dsu.find(j) again *after* the potential union to get the correct, potentially updated, root
            # and its corresponding max_idx.
            current_root_of_j = dsu.find(j)
            j = dsu.max_idx[current_root_of_j] + 1

            # The loop condition 'while j <= R' handles cases where the jump exceeds R
            # or where the max_idx is N (leading to j becoming N+1, which is > R).

    # Check if the graph is connected after processing all queries.
    # A connected graph with V vertices has an MST with V-1 edges.
    total_vertices = N + Q

    # If the number of edges successfully added to the MST is equal to total_vertices - 1,
    # it means all vertices are connected in a single component.
    if mst_edges == total_vertices - 1:
        print(mst_cost)
    else:
        # If we couldn't form an MST with V-1 edges, the graph is disconnected.
        print(-1)

# Call the solve function to run the program
solve()
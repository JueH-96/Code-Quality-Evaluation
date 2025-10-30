import heapq
import sys

# Set recursion depth limit to handle potential deep recursion in segment tree traversal
# Max depth is roughly log2(MAX_COORD_VALUE) + path length in collected vertices.
# MaxCoord is 2N. log2(4e5) is around 19. Path length can be up to N.
# A value around N + buffer should be safe for segment tree recursion stack depth.
sys.setrecursionlimit(2000005)

# Read N
N = int(sys.stdin.readline())

# Read W (0-indexed internally)
W = list(map(int, sys.stdin.readline().split()))

# Read L and R (0-indexed internally, but store 1-based values from input)
L = [0] * N
R = [0] * N

# Maximum coordinate value is 2N as per constraints
MAX_COORD_VALUE = 2 * N

for i in range(N):
    L[i], R[i] = map(int, sys.stdin.readline().split())
    # No need to track max coordinate from input, constraint says up to 2N

# Store initial leaf lists for Segment Trees (1-based coordinate indexing)
# leaves_L[coord] stores list of vertex indices i where L[i] == coord
initial_leaves_L = [[] for _ in range(MAX_COORD_VALUE + 1)]
# leaves_R[coord] stores list of vertex indices i where R[i] == coord
initial_leaves_R = [[] for _ in range(MAX_COORD_VALUE + 1)]

for i in range(N):
    # Store 0-indexed vertex index i
    initial_leaves_L[L[i]].append(i)
    initial_leaves_R[R[i]].append(i)

# Function to perform query and clear on segment tree leaves
# Modifies the 'current_leaves' list passed by reference
def query_and_clear(current_leaves, query_l, query_r, node_l, node_r):
    """
    Recursively queries the segment tree leaf lists and clears visited entries.
    This function operates on a specific copy of the leaf lists for one Dijkstra run.
    Args:
        current_leaves: The list of lists representing leaf data (modified).
        query_l, query_r: The coordinate range being queried (1-based).
        node_l, node_r: The coordinate range of the current segment tree node (1-based).
    Returns:
        A list of vertex indices collected from the leaves in the query range.
    """
    # If the current node's range is outside the query range
    if node_r < query_l or node_l > query_r:
        return []

    # If the current node is a leaf (covers a single coordinate)
    if node_l == node_r:
        # If this leaf's coordinate is within the query range
        if query_l <= node_l <= query_r:
            # Collect vertices from this leaf and clear the list
            collected = current_leaves[node_l]
            current_leaves[node_l] = [] # Clear the list at this leaf
            return collected
        else:
            return [] # Leaf is outside query range

    # If the current node is an internal node
    mid = node_l + (node_r - node_l) // 2

    # Recursively query children and collect results
    collected = []
    collected.extend(query_and_clear(current_leaves, query_l, query_r, node_l, mid))
    collected.extend(query_and_clear(current_leaves, query_l, query_r, mid + 1, node_r))

    return collected

# Read Q
Q = int(sys.stdin.readline())

# Process each query
for _ in range(Q):
    s_query, t_query = map(int, sys.stdin.readline().split())
    # Convert to 0-indexed vertex indices
    s = s_query - 1
    t = t_query - 1

    # Initialize Dijkstra's data structures for the current query
    dist = [float('inf')] * N # Minimum path weight from s
    # visited array is implicitly handled by the priority queue and dist updates
    # and explicitly by filtering collected neighbors.
    visited = [False] * N # Keep track of vertices finalized by Dijkstra

    # Min-priority queue stores (current_path_weight, vertex_index)
    pq = []

    # Dijkstra starts from s
    # The path weight to s is just its own weight W[s]
    dist[s] = W[s]
    heapq.heappush(pq, (dist[s], s))

    # Create fresh copies of leaf lists for segment trees for this query
    # This allows query_and_clear to modify the lists without affecting other queries
    # O(N) time complexity for copying lists per query
    current_leaves_L = [list(lst) for lst in initial_leaves_L]
    current_leaves_R = [list(lst) for lst in initial_leaves_R]

    # Run Dijkstra
    while pq:
        # Get vertex u with the smallest distance found so far
        d, u = heapq.heappop(pq)

        # If u is already visited (distance finalized), skip
        if visited[u]:
            continue

        # Mark u as visited (distance finalized)
        visited[u] = True
        # The distance d popped from PQ is the minimum distance to u.
        # dist[u] is now finalized. We don't strictly need to set dist[u] = d here
        # because we already set it before pushing to PQ and check `d > dist[u]`
        # if using a non-visited check, but using `visited` array is clearer.

        # If we reached the target vertex t, we found the shortest path
        if u == t:
            break # Exit Dijkstra loop for this query

        # Explore neighbors v of u.
        # An edge (u, v) exists iff [L_u, R_u] and [L_v, R_v] are disjoint.
        # This means R_u < L_v OR R_v < L_u.

        # Find potential neighbors v satisfying L_v > R_u.
        # These vertices v have L_v in the range [R_u + 1, MAX_COORD_VALUE].
        # Query the segment tree based on L coordinates.
        # The query range is [R[u] + 1, MAX_COORD_VALUE]. Note R[u] is 1-based.
        neighbors1 = query_and_clear(current_leaves_L, R[u] + 1, MAX_COORD_VALUE, 1, MAX_COORD_VALUE)

        # Process collected potential neighbors from the first condition
        for v in neighbors1:
            # Check if v is already visited in this Dijkstra run
            # We only relax edges to unvisited vertices
            if not visited[v]:
                # If v is unvisited, there's a potential edge (u, v).
                # The new path weight to v via u is dist[u] + W_v.
                # 'd' is the minimum distance found so far to u (which is dist[u] as u is popped).
                new_dist_v = d + W[v]

                # If this path provides a shorter distance to v than previously found
                if new_dist_v < dist[v]:
                    dist[v] = new_dist_v # Update minimum distance to v
                    # Add v to the priority queue with the new shorter distance
                    heapq.heappush(pq, (dist[v], v))

        # Find potential neighbors v satisfying R_v < L_u.
        # These vertices v have R_v in the range [1, L_u - 1].
        # Query the segment tree based on R coordinates.
        # The query range is [1, L[u] - 1]. Note L[u] is 1-based.
        neighbors2 = query_and_clear(current_leaves_R, 1, L[u] - 1, 1, MAX_COORD_VALUE)

        # Process collected potential neighbors from the second condition
        for v in neighbors2:
             # Check if v is already visited in this Dijkstra run
             # We only relax edges to unvisited vertices
             if not visited[v]:
                # If v is unvisited, there's a potential edge (u, v).
                # The new path weight to v via u is dist[u] + W_v.
                # 'd' is the minimum distance found so far to u.
                new_dist_v = d + W[v]

                # If this path provides a shorter distance to v than previously found
                if new_dist_v < dist[v]:
                    dist[v] = new_dist_v # Update minimum distance to v
                    # Add v to the priority queue with the new shorter distance
                    heapq.heappush(pq, (dist[v], v))

    # After Dijkstra finishes for this query (either found t or PQ is empty)
    # The minimum distance to t is stored in dist[t]
    if dist[t] == float('inf'):
        # If dist[t] is still infinity, t is unreachable from s
        print(-1)
    else:
        # Otherwise, print the minimum path weight
        print(dist[t])
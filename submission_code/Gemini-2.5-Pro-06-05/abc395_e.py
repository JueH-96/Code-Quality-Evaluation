import sys
import heapq

def solve():
    """
    Solves the shortest path problem on a graph with edge reversal operations.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem parameters
    # N: number of vertices, M: number of edges, X: cost to reverse all edges
    try:
        N, M, X = map(int, input().split())
    except ValueError:
        # Handles empty input lines that might occur in some environments
        return

    # The state graph has 2*N vertices.
    # This is because for each vertex v, we can be in one of two states:
    # 1. The graph is in its original (forward) direction.
    # 2. The graph has all its edges reversed.
    #
    # We map these states to integer nodes:
    # - Vertices 0 to N-1 represent (vertex i+1, forward_graph).
    # - Vertices N to 2N-1 represent (vertex i+1, reversed_graph).
    # So, for a 1-indexed vertex v from the input:
    # - State (v, forward) is node v-1.
    # - State (v, reversed) is node (v-1) + N.
    num_nodes = 2 * N
    adj = [[] for _ in range(num_nodes)]

    # Build the adjacency list for the state graph from the M input edges
    for _ in range(M):
        u, v = map(int, input().split())
        # Convert to 0-indexed vertices
        u_idx, v_idx = u - 1, v - 1

        # Edge for moving in the forward graph: u -> v
        # This corresponds to a transition from state (u, forward) to (v, forward)
        # with a cost of 1 (a single move).
        # In our node numbering, this is an edge from u_idx to v_idx.
        adj[u_idx].append((v_idx, 1))

        # Edge for moving in the reversed graph: v -> u
        # This corresponds to a transition from state (v, reversed) to (u, reversed)
        # with a cost of 1.
        # In our node numbering, this is an edge from (v_idx + N) to (u_idx + N).
        adj[v_idx + N].append((u_idx + N, 1))

    # Add edges representing the "reverse all edges" operation.
    # This operation has a cost of X and transitions between the forward and reversed states
    # for the same vertex.
    for i in range(N):
        # Transition from (i, forward) to (i, reversed)
        # In our node numbering: i -> i + N, cost X
        adj[i].append((i + N, X))

        # Transition from (i, reversed) to (i, forward)
        # In our node numbering: i + N -> i, cost X
        adj[i + N].append((i, X))

    # --- Dijkstra's Algorithm ---
    # We want to find the shortest path from the start state to any of the end states.
    # Start state: (vertex 1, forward) -> node 0
    # End states: (vertex N, forward) or (vertex N, reversed) -> node N-1 or node 2N-1

    # `dist[i]` will store the minimum cost to reach state graph node i.
    # Initialize all distances to infinity.
    inf = float('inf')
    dist = [inf] * num_nodes

    # We start at vertex 1 in the forward graph state.
    # The node for (vertex 1, forward) is 0. Cost to reach start is 0.
    start_node = 0
    dist[start_node] = 0

    # The priority queue stores tuples of (cost, node).
    # We use a min-heap, so the element with the smallest cost is always at the top.
    pq = [(0, start_node)]

    while pq:
        # Get the node with the smallest known distance from the source
        d, u = heapq.heappop(pq)

        # If we have already found a shorter path to u, this entry is stale. Skip.
        if d > dist[u]:
            continue

        # Explore neighbors of u
        for v, weight in adj[u]:
            # If we found a shorter path to v through u, update its distance
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                # Add the updated path information to the priority queue
                heapq.heappush(pq, (dist[v], v))

    # The goal is to reach vertex N, regardless of the graph's direction state.
    # Node for (vertex N, forward) is N-1.
    # Node for (vertex N, reversed) is (N-1) + N = 2*N-1.
    target_node_f = N - 1
    target_node_r = 2 * N - 1

    # The minimum cost to reach vertex N is the minimum of the costs to reach
    # either of its corresponding states in our state graph.
    result = min(dist[target_node_f], dist[target_node_r])

    print(result)

solve()
# YOUR CODE HERE
import heapq
import sys

def solve():
    # Read input: N vertices, M edges, cost X to reverse edges
    N, M, X = map(int, sys.stdin.readline().split())

    # Initialize adjacency lists for the original graph and the reversed graph
    # adj[u] stores neighbors v such that there is an edge u -> v
    # rev_adj[v] stores neighbors u such that there is an edge u -> v
    adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)]

    # Populate adjacency lists from input edges
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Convert to 0-based indexing for internal representation
        u -= 1
        v -= 1
        adj[u].append(v)
        rev_adj[v].append(u)

    # The problem can be modeled as a shortest path problem on a state graph.
    # A state is defined by (current_vertex, edge_direction).
    # Edge directions can be original (0) or reversed (1).
    # The state space graph has 2N vertices.
    # State indices 0 to N-1 represent states (vertex_idx, 0) for original direction.
    # State indices N to 2N-1 represent states (vertex_idx, 1) for reversed direction.
    # State for vertex i with original direction corresponds to index i.
    # State for vertex i with reversed direction corresponds to index i + N.
    
    # Initialize distances to all states as infinity
    dist = [float('inf')] * (2 * N)
    
    # The starting state is vertex 1 (index 0) with original edge directions.
    # The state index for (vertex 0, direction 0) is 0.
    start_state_idx = 0 
    dist[start_state_idx] = 0
    
    # Priority queue for Dijkstra's algorithm. Stores tuples of (current_cost, state_index).
    # Min-heap based on cost.
    pq = [(0, start_state_idx)] 

    while pq:
        # Extract the state with the minimum cost found so far
        current_cost, u_idx = heapq.heappop(pq)

        # If we found a shorter path to this state already, skip processing
        if current_cost > dist[u_idx]:
            continue 

        # Determine the current vertex index and edge direction based on the state index u_idx
        if u_idx < N:
            # Current state is (u_node_idx, 0): At vertex u_node_idx, edges are in original direction.
            u_node_idx = u_idx
            
            # Explore neighbors via two types of operations:

            # Option 1: Move along a directed edge u -> v
            # This edge must exist in the current graph configuration (original directions).
            # Neighbors v are found using adj[u_node_idx].
            for v_node_idx in adj[u_node_idx]:
                # The target state is (v_node_idx, 0). Its state index is v_node_idx.
                v_state_idx = v_node_idx 
                new_cost = current_cost + 1 # Cost of moving along an edge is 1
                # If this path is shorter than the known shortest path to v_state_idx:
                if new_cost < dist[v_state_idx]:
                    dist[v_state_idx] = new_cost # Update distance
                    heapq.heappush(pq, (new_cost, v_state_idx)) # Add to priority queue

            # Option 2: Reverse all edges
            # Stay at vertex u_node_idx, but change edge direction to reversed.
            # The target state is (u_node_idx, 1). Its state index is u_node_idx + N.
            v_state_idx = u_node_idx + N 
            new_cost = current_cost + X # Cost of reversing edges is X
            # If this path is shorter:
            if new_cost < dist[v_state_idx]:
                dist[v_state_idx] = new_cost # Update distance
                heapq.heappush(pq, (new_cost, v_state_idx)) # Add to priority queue

        else: # u_idx >= N
            # Current state is (u_node_idx, 1): At vertex u_node_idx, edges are reversed.
            u_node_idx = u_idx - N

            # Explore neighbors:

            # Option 1: Move along a directed edge u -> v in the reversed graph
            # An edge u -> v exists in the reversed graph if and only if
            # the edge v -> u exists in the original graph.
            # We find such neighbors v by looking into rev_adj[u_node_idx].
            for v_node_idx in rev_adj[u_node_idx]:
                # The target state is (v_node_idx, 1), as we move to vertex v and edges remain reversed.
                # Its state index is v_node_idx + N.
                v_state_idx = v_node_idx + N 
                new_cost = current_cost + 1 # Cost of move is 1
                # If this path is shorter:
                if new_cost < dist[v_state_idx]:
                    dist[v_state_idx] = new_cost # Update distance
                    heapq.heappush(pq, (new_cost, v_state_idx)) # Add to priority queue

            # Option 2: Reverse edges back to original direction
            # Stay at vertex u_node_idx, change edge direction back to original.
            # The target state is (u_node_idx, 0). Its state index is u_node_idx.
            v_state_idx = u_node_idx 
            new_cost = current_cost + X # Cost of reverse is X
            # If this path is shorter:
            if new_cost < dist[v_state_idx]:
                dist[v_state_idx] = new_cost # Update distance
                heapq.heappush(pq, (new_cost, v_state_idx)) # Add to priority queue

    # The target vertex is N. Its 0-based index is N-1.
    target_node_idx = N - 1
    
    # The final minimum cost is the minimum cost to reach vertex N
    # either with original edges (state index N-1)
    # or with reversed edges (state index (N-1) + N = 2N - 1).
    
    final_cost = min(dist[target_node_idx], dist[target_node_idx + N])

    # The problem statement guarantees that vertex N is reachable from vertex 1.
    # Thus, final_cost will be a finite value (not infinity).
    print(final_cost)

# Execute the solve function to run the program
solve()
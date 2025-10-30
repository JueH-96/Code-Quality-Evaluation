import sys
import heapq
from collections import defaultdict

# Setting recursion depth for potentially deep graphs typical in competitive programming
# The default limit (e.g., 1000) might be too small for N up to 2*10^5.
try:
    # Set recursion depth limit higher than default.
    # Value 200010 chosen based on max N = 2e5 plus a small buffer.
    sys.setrecursionlimit(200010) 
except Exception as e:
    # Optional: print warning if setting fails, e.g., due to OS limits.
    # This might happen on some systems/environments.
    # print(f"Warning: Failed to set recursion depth limit: {e}", file=sys.stderr)
    pass 

def solve():
    # Read input: N = number of cities, M = number of roads
    N, M = map(int, sys.stdin.readline().split())

    # adj stores the graph using adjacency lists: node -> list of (neighbor, cost, road_index)
    adj = defaultdict(list)
    # edges list stores info about each road for easy lookup later
    edges = [] 
    for i in range(M):
        # Read road details: connects cities A_i and B_i with cost C_i
        u, v, c = map(int, sys.stdin.readline().split())
        # Convert to 0-based indexing for internal use
        u -= 1 
        v -= 1
        # Store 1-based road index as required for output logic
        adj[u].append((v, c, i + 1)) 
        adj[v].append((u, c, i + 1))
        # Store edge details: 0-based nodes u, v, cost c, 1-based road ID
        edges.append({'u': u, 'v': v, 'c': c, 'id': i + 1})

    # Dijkstra's algorithm implementation
    def dijkstra(start_node):
        # Initialize distances: infinite for all nodes except start_node (0 distance)
        # Using defaultdict simplifies handling unvisited nodes (defaulting to infinity)
        dist = defaultdict(lambda: float('inf'))
        dist[start_node] = 0
        # Priority queue stores (current_distance, node) tuples, min-heap based on distance
        pq = [(0, start_node)]

        while pq:
            # Extract node with smallest distance from priority queue
            d, u = heapq.heappop(pq)

            # Optimization: If we found a shorter path already, skip processing this element
            if d > dist[u]:
                continue

            # Iterate through neighbors of node u
            for v, c, _ in adj[u]: # _ ignores the road_index stored in adj list entry
                # Relaxation step: If path through u to v is shorter than current known distance to v
                if dist[u] + c < dist[v]:
                    # Update distance to v
                    dist[v] = dist[u] + c
                    # Add/update v in the priority queue with new shorter distance
                    heapq.heappush(pq, (dist[v], v))
        
        # Return dictionary mapping nodes to their shortest distance from start_node
        return dist

    # Compute shortest distances from node 0 (city 1) to all other nodes
    dist1 = dijkstra(0)
    # Compute shortest distances from node N-1 (city N) to all other nodes
    # This is equivalent to finding shortest paths TO node N-1 in the original graph
    distN = dijkstra(N - 1)

    # The overall shortest distance from city 1 to city N
    shortest_dist = dist1[N - 1]
    
    # Problem statement guarantees city N is reachable from city 1 when all roads are passable.
    # Thus, shortest_dist is guaranteed to be finite.

    # Construct the Shortest Path Graph (SPG)
    # SPG contains only edges that lie on at least one shortest path from node 0 to node N-1
    # spg_adj maps node u -> dictionary {neighbor v: road_index} for edges in SPG
    spg_adj = defaultdict(dict) 
    
    for edge_info in edges:
        u, v, c, road_idx = edge_info['u'], edge_info['v'], edge_info['c'], edge_info['id']
        
        # An edge (u, v) with cost c is on *some* shortest path if either:
        # 1. The path uses the edge in direction u -> v: dist(1, u) + c + dist(v, N) = shortest_dist(1, N)
        # 2. The path uses the edge in direction v -> u: dist(1, v) + c + dist(u, N) = shortest_dist(1, N)
        
        # Note: dist(x, N) is computed by Dijkstra from N, which is distN[x].
        
        is_on_sp = False
        # Check condition 1: path uses edge as u -> v
        # Ensure distances are finite before arithmetic to avoid potential issues with inf + finite = inf comparison.
        if dist1[u] != float('inf') and distN[v] != float('inf') and dist1[u] + c + distN[v] == shortest_dist:
            is_on_sp = True
        
        # Check condition 2: path uses edge as v -> u
        # This check is only needed if condition 1 failed.
        if not is_on_sp and dist1[v] != float('inf') and distN[u] != float('inf') and dist1[v] + c + distN[u] == shortest_dist:
             is_on_sp = True

        # If the edge (corresponding to road_idx) is on any shortest path
        if is_on_sp:
             # Add an UNDIRECTED edge to the SPG. Representing it by adding entries for both u->v and v->u.
             # The check `v not in spg_adj[u]` ensures we add the edge only once.
             # It implicitly handles the case where both conditions might be true (e.g., if c=0, which is not the case here).
             if v not in spg_adj[u]: 
                 spg_adj[u][v] = road_idx
                 spg_adj[v][u] = road_idx

    # Find critical edges using DFS based bridge finding algorithm on the SPG.
    # An edge is critical if it's a bridge in the SPG AND its removal disconnects node 0 from node N-1.
    visited_rec = set() # Set of visited nodes during DFS
    disc_rec = {}  # Dictionary storing discovery time of node u
    low_rec = {}   # Dictionary storing lowest discovery time reachable from node u (including itself and via back-edges from its subtree)
    timer_rec = 0 # Global timer for assigning discovery times
    critical_roads_rec = set() # Set to store 1-based indices of critical roads identified
    
    # foundN_subtree[u] becomes true if node N-1 is reachable within the DFS subtree rooted at u
    foundN_subtree = defaultdict(bool) 

    # Recursive DFS function for bridge finding and checking criticality
    def dfs_recursive(u, p): # u: current node, p: parent node in DFS tree
        nonlocal timer_rec # Allow modification of the global timer
        visited_rec.add(u)
        # Initialize discovery time and low-link value for node u
        disc_rec[u] = low_rec[u] = timer_rec
        timer_rec += 1
        
        # Check if the current node u is the target node N-1
        current_foundN = (u == N - 1) 

        # Explore neighbors of u in the SPG
        if u in spg_adj: 
            for v, road_idx in spg_adj[u].items():
                # Skip the edge leading back to the parent node to avoid trivial cycles
                if v == p: continue

                # Case 1: Neighbor v is already visited
                if v in visited_rec:
                    # This indicates a back-edge (or cross-edge to an already visited part of tree)
                    # Update low-link value of u using the discovery time of v
                    low_rec[u] = min(low_rec[u], disc_rec[v])
                
                # Case 2: Neighbor v is not visited yet
                else: 
                    # This is a tree edge in the DFS tree
                    # Recursively explore the subtree rooted at v
                    child_foundN = dfs_recursive(v, u)
                    # After the recursive call returns, update low-link value of u
                    # based on the lowest discovery time reachable from v's subtree
                    low_rec[u] = min(low_rec[u], low_rec[v])

                    # Check if the edge (u, v) forms a bridge
                    # Condition: The lowest reachable node from v's subtree (low_rec[v])
                    # has a discovery time strictly greater than the discovery time of u.
                    # This means there's no back-edge from v's subtree to u or its ancestors.
                    if low_rec[v] > disc_rec[u]: 
                        # If (u, v) is a bridge AND it lies on a path towards the target N-1
                        # (i.e., N-1 is reachable from v downwards in the DFS tree)
                        if child_foundN: 
                            # Mark the corresponding road as critical
                            critical_roads_rec.add(road_idx)
                    
                    # Propagate the "found N" status up the DFS tree:
                    # If N-1 was found in the subtree of v, it means N-1 is reachable from u downwards.
                    current_foundN = current_foundN or child_foundN
        
        # Store the result for node u: whether N-1 is reachable from this subtree.
        foundN_subtree[u] = current_foundN 
        # Return the status for use by the parent node in the recursion.
        return current_foundN

    # Start the DFS from node 0 (city 1). This is valid because N >= 2.
    if N > 1:
        # Node 0 must be part of the SPG if a shortest path exists (which is guaranteed).
        # Start DFS from node 0. The parent is set to -1 (an invalid node index) to indicate root.
        dfs_recursive(0, -1) 

    # Prepare the final output strings based on whether each road was marked critical.
    results = []
    for i in range(1, M + 1): # Iterate through road indices 1 to M
        if i in critical_roads_rec:
           results.append("Yes") # Road i is critical
        else:
           results.append("No")  # Road i is not critical
    
    # Print all results, each on a new line, as required by the output format.
    print("
".join(results))

# Execute the main function to solve the problem
solve()
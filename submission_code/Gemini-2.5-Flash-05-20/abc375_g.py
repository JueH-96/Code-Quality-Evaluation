# Standard library imports
import heapq
import sys

# Define a constant for infinity, suitable for distance calculations.
# Using float('inf') is standard and safe for comparisons.
INF = float('inf')

def dijkstra_distances(start_node, N, adj_list):
    """
    Computes the shortest distances from a start_node to all other nodes in the graph.
    
    Args:
        start_node (int): The starting node for Dijkstra's algorithm.
        N (int): The total number of nodes in the graph (1 to N).
        adj_list (list): Adjacency list representation of the graph.
                         adj_list[u] contains tuples (v, weight, edge_idx).
    
    Returns:
        list: A list where distances[i] is the shortest distance from start_node to node i.
              Returns INF for unreachable nodes.
    """
    distances = [INF] * (N + 1)
    distances[start_node] = 0
    # Priority queue stores (distance, node). Min-heap ensures shortest distance is popped first.
    pq = [(0, start_node)] 
    
    while pq:
        d, u = heapq.heappop(pq)
        
        # If we have already found a shorter path to u, skip this entry.
        # This handles cases where nodes are pushed multiple times with different distances.
        if d > distances[u]:
            continue
            
        # Explore neighbors of u
        for v, weight, _ in adj_list[u]: # `_` is original_edge_index, not needed for distances
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))
    return distances

def count_shortest_paths(start_node, N, adj_list, distances):
    """
    Computes the number of shortest paths from a start_node to all other nodes,
    given pre-computed shortest distances. This is done by processing nodes
    in topological order of the Shortest Path DAG.
    
    Args:
        start_node (int): The starting node.
        N (int): The total number of nodes.
        adj_list (list): Adjacency list of the graph.
        distances (list): Shortest distances from start_node, as computed by dijkstra_distances.
    
    Returns:
        list: A list where num_paths[i] is the number of shortest paths from start_node to node i.
              Returns 0 for unreachable nodes or nodes not on any shortest path from start_node.
    """
    num_paths = [0] * (N + 1)
    num_paths[start_node] = 1 # There's one way to reach the start node (itself)
    
    # Sort nodes by their shortest distance from the start_node.
    # This order ensures that when processing node `u`, all `num_paths` values
    # for nodes that come before `u` in a shortest path are already finalized.
    # Filter out unreachable nodes to avoid issues with INF distances during sorting.
    nodes_reachable_from_start = [i for i in range(1, N + 1) if distances[i] != INF]
    sorted_nodes_by_distance = sorted(nodes_reachable_from_start, key=lambda x: distances[x])

    for u in sorted_nodes_by_distance:
        # If num_paths[u] is 0, it means 'u' is not reachable from `start_node` via a shortest path.
        # This condition helps ensure we only propagate counts from nodes that are part of a shortest path.
        if num_paths[u] == 0: 
            continue

        # Iterate over neighbors `v` of `u`
        for v, weight, _ in adj_list[u]:
            # If the path through `u` to `v` (u -> v) forms a shortest path from `start_node` to `v`
            if distances[u] + weight == distances[v]:
                num_paths[v] += num_paths[u]
    return num_paths

def main():
    # Optimize input reading for speed
    input = sys.stdin.readline
    N, M = map(int, input().split())

    # Adjacency list: adj_list[u] contains (neighbor_v, weight, original_edge_index)
    adj_list = [[] for _ in range(N + 1)]
    # Store original edge details to easily retrieve A, B, C for each road index
    original_edges = [] 

    for i in range(M):
        A, B, C = map(int, input().split())
        adj_list[A].append((B, C, i))
        adj_list[B].append((A, C, i)) # Roads are bidirectional
        original_edges.append((A, B, C)) # Store (A, B, C) for road i

    # Step 1: Compute shortest distances from city 1 to all other cities
    dist1 = dijkstra_distances(1, N, adj_list)
    # Step 2: Compute the number of shortest paths from city 1 to all other cities
    num_paths1 = count_shortest_paths(1, N, adj_list, dist1)

    # Step 3: Compute shortest distances from city N to all other cities
    # This is equivalent to running Dijkstra from N on the reversed graph,
    # but since edges are bidirectional, it's the same as running on the original graph.
    distN = dijkstra_distances(N, N, adj_list)
    # Step 4: Compute the number of shortest paths from all cities to city N
    num_pathsN = count_shortest_paths(N, N, adj_list, distN)

    # Step 5: Get the shortest distance from city 1 to city N using all roads
    D_all = dist1[N]

    # Step 6: Get the total number of shortest paths from city 1 to city N
    # The problem guarantees N is reachable from 1, so total_shortest_paths will be > 0.
    total_shortest_paths = num_paths1[N]

    # Initialize results for each road
    results = ["No"] * M

    # Step 7: For each original road, determine if removing it changes the shortest distance
    for i in range(M):
        A_i, B_i, C_i = original_edges[i]
        
        current_edge_contributing_paths = 0

        # Check if the path segment 1 -> ... -> A_i -> B_i -> ... -> N forms a shortest path
        # Ensure A_i and B_i are reachable from 1 and N respectively to avoid INF arithmetic issues.
        # This check is crucial because (dist1[A_i] + C_i + distN[B_i]) would be INF if any of A_i or B_i are unreachable.
        if dist1[A_i] != INF and distN[B_i] != INF:
            if dist1[A_i] + C_i + distN[B_i] == D_all:
                # Add the number of shortest paths that pass through A_i then B_i
                current_edge_contributing_paths += num_paths1[A_i] * num_pathsN[B_i]
                
        # Check if the path segment 1 -> ... -> B_i -> A_i -> ... -> N forms a shortest path
        if dist1[B_i] != INF and distN[A_i] != INF:
            if dist1[B_i] + C_i + distN[A_i] == D_all:
                # Add the number of shortest paths that pass through B_i then A_i
                current_edge_contributing_paths += num_paths1[B_i] * num_pathsN[A_i]
        
        # If the sum of all shortest paths that use this specific road (in either direction)
        # is equal to the total number of shortest paths from 1 to N,
        # then this road is essential for achieving D_all. Removing it will make the path longer.
        if current_edge_contributing_paths == total_shortest_paths:
            results[i] = "Yes"

    # Step 8: Print results
    for res in results:
        sys.stdout.write(res + '
')

if __name__ == '__main__':
    main()
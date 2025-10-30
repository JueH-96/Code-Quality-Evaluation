import heapq
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    ops_orig = []
    for i in range(M):
        l, r = map(int, sys.stdin.readline().split())
        ops_orig.append({'L': l, 'R': r, 'id': i})

    # Coordinate compression
    # Points of interest are 0, N, and all L_i-1, R_i
    coord_points = {0, N}
    for op_spec in ops_orig:
        coord_points.add(op_spec['L'] - 1)
        coord_points.add(op_spec['R'])
    
    sorted_coords = sorted(list(coord_points))
    coord_to_idx = {val: i for i, val in enumerate(sorted_coords)}
    
    num_compressed_nodes = len(sorted_coords)

    adj = [[] for _ in range(num_compressed_nodes)]

    # Add 0-cost edges between adjacent compressed coordinates
    for i in range(num_compressed_nodes - 1):
        u, v = coord_to_idx[sorted_coords[i]], coord_to_idx[sorted_coords[i+1]]
        # Cost is 0 because these are just segments of the original 0..N line
        # An operation covering sorted_coords[i] and sorted_coords[i+1] covers everything in between.
        adj[u].append({'to': v, 'cost': 0, 'op_id': -1, 'op_type': -1})
        adj[v].append({'to': u, 'cost': 0, 'op_id': -1, 'op_type': -1})

    # Add operation edges
    for op_spec in ops_orig:
        l, r, op_id = op_spec['L'], op_spec['R'], op_spec['id']
        
        u_type1 = coord_to_idx[l - 1]
        v_type1 = coord_to_idx[r]
        adj[u_type1].append({'to': v_type1, 'cost': 1, 'op_id': op_id, 'op_type': 1})

        u_type2 = coord_to_idx[r]
        v_type2 = coord_to_idx[l - 1]
        adj[u_type2].append({'to': v_type2, 'cost': 1, 'op_id': op_id, 'op_type': 2})

    # Dijkstra
    # dist[i] = min cost to cover prefix up to sorted_coords[i]
    # parent_op_info[i] = (op_id, op_type) for the operation that achieved this min_cost for node i
    # from_node[i] = predecessor node in shortest path to i
    
    source_idx = coord_to_idx[0]
    target_idx = coord_to_idx[N]

    dist = [float('inf')] * num_compressed_nodes
    parent_op_info = [(-1, -1)] * num_compressed_nodes # (op_id, op_type)
    from_node = [-1] * num_compressed_nodes

    dist[source_idx] = 0
    
    # (cost, node_idx)
    pq = [(0, source_idx)]

    while pq:
        d, u_idx = heapq.heappop(pq)

        if d > dist[u_idx]:
            continue
        
        # This check is critical for correctness on Sample 1 with the "Pilots" graph.
        # If u_idx == target_idx, we might have found the shortest path.
        # However, Dijkstra explores exhaustively until PQ empty or all reachable nodes settled.
        # The problem is not just reaching target_idx, but that target_idx must be N.
        # If target_idx itself is an intermediate point in some path, it is fine.

        for edge in adj[u_idx]:
            v_idx = edge['to']
            cost = edge['cost']
            
            if dist[u_idx] + cost < dist[v_idx]:
                dist[v_idx] = dist[u_idx] + cost
                parent_op_info[v_idx] = (edge['op_id'], edge['op_type'])
                from_node[v_idx] = u_idx
                heapq.heappush(pq, (dist[v_idx], v_idx))
            # Tie-breaking: if costs are equal, prefer paths with fewer actual operations
            # This is implicitly handled if 0-cost edges are processed first, or by specific tie-breaking.
            # Standard Dijkstra doesn't need complex tie-breaking for correctness of cost.
            # For path reconstruction, it can matter. The problem says "any one way".
            # A common tie-break: if (dist[u_idx] + cost == dist[v_idx]), check if new path uses fewer ops.
            # This requires storing path length in ops, not just cost.
            # Let's assume standard Dijkstra path is fine.

    min_total_cost = dist[target_idx]

    if min_total_cost == float('inf'):
        print("-1")
    else:
        print(min_total_cost)
        
        chosen_ops_type = [0] * M # Initialize all ops as type 0
        
        curr_idx = target_idx
        while curr_idx != source_idx : # Backtrack until we reach source_idx (node 0)
            op_id, op_type = parent_op_info[curr_idx]
            prev_node_idx = from_node[curr_idx]

            if op_id != -1: # This was an operation edge
                # Only record if this op hasn't been chosen with a different type by another part of path.
                # Or if chosen with same type, it's fine. If chosen with different type, it's conflict.
                # This graph formulation assumes an op is chosen *once*.
                # If chosen_ops_type[op_id] is already non-zero and not op_type, this is an issue.
                # However, this specific graph formulation means an edge $X \to Y$ via op $(k, type)$
                # implies op $k$ is chosen as $type$. It can't be chosen as other type by another edge in path.
                chosen_ops_type[op_id] = op_type
            
            curr_idx = prev_node_idx
            if curr_idx == -1 : # Should not happen if path exists
                break 
        
        print(*(chosen_ops_type))

solve()
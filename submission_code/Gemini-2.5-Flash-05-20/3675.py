import collections

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1  # Number of nodes (0 to n-1)
        
        # Step 1: Initialize data structures and calculate initial total weight
        total_sum_kept = 0
        initial_degrees = collections.defaultdict(int)
        # adj_edges stores (weight, neighbor_node, edge_index)
        # edge_index is crucial for uniquely identifying edges later
        adj_edges = collections.defaultdict(list) 

        for edge_idx, (u, v, w) in enumerate(edges):
            total_sum_kept += w
            initial_degrees[u] += 1
            initial_degrees[v] += 1
            adj_edges[u].append((w, v, edge_idx))
            adj_edges[v].append((w, u, edge_idx))

        # Step 2: For each node, sort its incident edges by weight
        # and identify candidate edges for removal
        candidate_removals = [] # List of (weight, u, v, edge_idx) tuples

        # Iterate through all possible node IDs (0 to n-1)
        # A node might exist but have no edges if n is much larger than the number of nodes explicitly in 'edges'
        for node in range(n):
            if node not in initial_degrees:
                # This node has no edges, its degree is 0, which is <= k.
                # No need to process further for this node.
                continue

            adj_edges[node].sort() # Sort edges incident to this node by weight (ascending)
            
            # Calculate how many edges this node 'wants' to remove
            # from its cheapest end to satisfy its degree constraint
            num_to_remove_for_node = max(0, initial_degrees[node] - k)
            
            # Add these cheapest edges as candidates for removal
            # Each candidate is stored as (weight, one_endpoint, other_endpoint, original_edge_index)
            for i in range(num_to_remove_for_node):
                w, neighbor, edge_idx = adj_edges[node][i]
                candidate_removals.append((w, node, neighbor, edge_idx))
        
        # Step 3: Sort all collected candidate edges by weight in ascending order
        # This ensures we always consider the cheapest edge that some node wants to remove.
        candidate_removals.sort()

        # Step 4: Greedily remove edges
        removed_sum = 0
        # This dictionary will track the current degree of each node after removals.
        # It starts with initial degrees and gets updated.
        final_degrees = {node: initial_degrees[node] for node in initial_degrees}
        # This set stores the indices of edges that have already been marked for removal.
        # It prevents processing the same physical edge multiple times.
        removed_edge_ids = set()

        for weight, u, v, edge_idx in candidate_removals:
            # If this edge has already been processed and removed, skip it.
            # This happens if it was a candidate for two nodes, and processed by the other node first.
            if edge_idx in removed_edge_ids:
                continue 

            # Check if removing this edge is still necessary to satisfy the degree constraint
            # for *either* node u or node v.
            # An edge is necessary for removal if at least one of its endpoints
            # currently has a degree greater than k.
            # Since we process edges by increasing weight, this is the cheapest way
            # to reduce the degree for an over-connected node.
            
            # Use .get(node, 0) in case a node somehow ended up not in initial_degrees,
            # though for a connected tree it should be fine.
            u_current_degree = final_degrees.get(u, 0)
            v_current_degree = final_degrees.get(v, 0)

            if u_current_degree > k or v_current_degree > k:
                # This edge is needed for removal to help reduce degrees.
                removed_sum += weight
                final_degrees[u] -= 1
                final_degrees[v] -= 1
                removed_edge_ids.add(edge_idx)
            # Else (if u_current_degree <= k AND v_current_degree <= k):
            # This edge is no longer needed to satisfy degree constraints.
            # It means that other, cheaper (or equally cheap) edge removals have
            # already brought both endpoints' degrees to k or below.
            # So, we do NOT remove this edge, to maximize the sum of remaining weights.
        
        # The maximum possible sum of weights for the remaining edges
        # is the initial total sum minus the sum of weights of the edges we decided to remove.
        return total_sum_kept - removed_sum
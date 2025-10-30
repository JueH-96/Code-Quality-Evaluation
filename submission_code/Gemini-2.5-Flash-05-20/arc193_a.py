import heapq
import sys

# Set a high value for infinity. Path weights can sum up to 2*10^5 * 10^9 = 2*10^14.
INF = float('inf') 

# Max coordinate value for segment tree. R_i can go up to 2N. 2*2*10^5 = 4*10^5.
# Add a small buffer just in case, though 2N itself should be fine.
MAX_COORD_VALUE = 4 * 10**5 + 5 

class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Each node in the segment tree stores a (W_val, vertex_id) pair.
        # Initialize with (INF, -1) indicating no valid vertex.
        self.tree = [(INF, -1)] * (4 * size) 
        
    def _update_tree(self, node_idx, start, end, target_coord, value):
        """
        Updates the value at a specific target_coord in the segment tree.
        Propagates the minimum value up to parent nodes.
        """
        if start == end: # Leaf node reached
            self.tree[node_idx] = value
            return
        
        mid = (start + end) // 2
        if start <= target_coord <= mid:
            # Target coordinate is in the left child's range
            self._update_tree(2 * node_idx, start, mid, target_coord, value)
        else:
            # Target coordinate is in the right child's range
            self._update_tree(2 * node_idx + 1, mid + 1, end, target_coord, value)
        
        # After updating children, update the current node with the minimum of its children.
        self.tree[node_idx] = min(self.tree[2 * node_idx], self.tree[2 * node_idx + 1])

    def _query_tree(self, node_idx, start, end, query_start, query_end, visited):
        """
        Queries for the minimum (W_val, vertex_id) pair within a given range.
        `visited` array is used to ensure only unvisited vertices are considered.
        """
        # If the current segment tree node's range is outside the query range, return INF.
        if end < query_start or start > query_end:
            return (INF, -1)
        
        # Get the current minimum stored at this node.
        val, v_id = self.tree[node_idx]
        
        # If the vertex corresponding to this node's minimum is already visited,
        # it means this minimum is stale. We need to go deeper to find a valid one.
        if v_id != -1 and visited[v_id]:
            # If it's a leaf node and its only candidate is visited, there's no valid min in this leaf.
            if start == end:
                return (INF, -1)
            # If it's an internal node, recursively query children to find the true min among unvisited.
            mid = (start + end) // 2
            min_left = self._query_tree(2 * node_idx, start, mid, query_start, query_end, visited)
            min_right = self._query_tree(2 * node_idx + 1, mid + 1, end, query_start, query_end, visited)
            return min(min_left, min_right)
        
        # If the current segment tree node's range is completely within the query range AND its min is valid.
        # This means we found a valid minimum candidate for the queried range.
        if query_start <= start and end <= query_end:
            return (val, v_id) 
        
        # If the current segment tree node's range partially overlaps with the query range, recurse into children.
        mid = (start + end) // 2
        min_left = self._query_tree(2 * node_idx, start, mid, query_start, query_end, visited)
        min_right = self._query_tree(2 * node_idx + 1, mid + 1, end, query_start, query_end, visited)
        
        # Return the overall minimum from the children's results.
        return min(min_left, min_right)

    def update(self, coord, value):
        """Public update method to be called by Dijkstra."""
        self._update_tree(1, 1, self.size, coord, value)

    def query(self, query_start, query_end, visited):
        """Public query method to be called by Dijkstra."""
        if query_start > query_end: # Handle empty query range
            return (INF, -1)
        return self._query_tree(1, 1, self.size, query_start, query_end, visited)

def solve():
    N = int(sys.stdin.readline())
    # W is 1-indexed for convenience (W[0] is dummy)
    W = [0] + list(map(int, sys.stdin.readline().split())) 
    
    # intervals are 1-indexed (intervals[0] is dummy)
    intervals = [None] * (N + 1) 
    for i in range(1, N + 1):
        L, R = map(int, sys.stdin.readline().split())
        intervals[i] = (L, R)

    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        s, t = map(int, sys.stdin.readline().split())
        queries.append((s, t))

    # Initialize two global segment trees. They will be modified during Dijkstra
    # and then reverted for each query.
    st_L = SegmentTree(MAX_COORD_VALUE) # For L_v > R_u queries
    st_R = SegmentTree(MAX_COORD_VALUE) # For R_v < L_u queries

    # Pre-process initial minimum weights for each coordinate.
    # This prepares the initial state of the segment trees.
    initial_L_coords_data = {} # Maps L_coord -> (W_min, v_idx_min)
    initial_R_coords_data = {} # Maps R_coord -> (W_min, v_idx_min)

    for i in range(1, N + 1):
        L_i, R_i = intervals[i]
        
        # Update initial_L_coords_data with the minimum weight vertex for L_i
        current_min_L = initial_L_coords_data.get(L_i, (INF, -1))
        initial_L_coords_data[L_i] = min(current_min_L, (W[i], i))

        # Update initial_R_coords_data with the minimum weight vertex for R_i
        current_min_R = initial_R_coords_data.get(R_i, (INF, -1))
        initial_R_coords_data[R_i] = min(current_min_R, (W[i], i))

    # Populate the segment trees with the initial minimum values.
    for coord, val in initial_L_coords_data.items():
        st_L.update(coord, val)
    for coord, val in initial_R_coords_data.items():
        st_R.update(coord, val)

    def dijkstra(s, t):
        """
        Runs Dijkstra's algorithm from source s to target t.
        Modifies global segment trees and reverts changes afterwards.
        """
        dist = [INF] * (N + 1)
        dist[s] = W[s]
        
        pq = [(W[s], s)] # Min-priority queue: (current_path_weight, vertex)
        
        # `visited` array to keep track of vertices whose distance is finalized.
        visited = [False] * (N + 1)

        # Lists to store changes made to segment trees during this Dijkstra run.
        # Used to efficiently revert changes for the next query.
        changes_to_revert_L = []
        changes_to_revert_R = []

        while pq:
            current_path_weight, u = heapq.heappop(pq)

            if visited[u]:
                continue # Already found a shorter path or processed this vertex

            visited[u] = True
            
            if u == t: # Target reached, shortest path found
                # Revert all changes made to segment trees before returning.
                for coord, val in changes_to_revert_L:
                    st_L.update(coord, val)
                for coord, val in changes_to_revert_R:
                    st_R.update(coord, val)
                return current_path_weight

            # Get interval for current vertex u
            L_u, R_u = intervals[u]
            
            # When `u` is processed, it should no longer be considered as a candidate for neighbors.
            # "Remove" it from segment trees by setting its weight at its coordinate to INF.
            
            # Store old values to revert later
            # For `st_L`, the point `L_u` (start of interval `u`)
            changes_to_revert_L.append((L_u, st_L.query(L_u, L_u, [False]*(N+1)))) 
            # For `st_R`, the point `R_u` (end of interval `u`)
            changes_to_revert_R.append((R_u, st_R.query(R_u, R_u, [False]*(N+1))))

            st_L.update(L_u, (INF, -1))
            st_R.update(R_u, (INF, -1))
            
            # Find neighbors `v` where `R_v < L_u` (v's interval ends before u's begins)
            # Query `st_R` for minimum `W_v` in range `[1, L_u - 1]`.
            while True:
                min_w_v, v_candidate = st_R.query(1, L_u - 1, visited)
                
                if min_w_v == INF: # No more valid candidates in this range
                    break
                
                # Calculate new path cost to v_candidate
                new_cost_to_v = current_path_weight - W[u] + min_w_v
                
                # If this path is shorter, update distance and push to PQ
                if new_cost_to_v < dist[v_candidate]:
                    dist[v_candidate] = new_cost_to_v
                    heapq.heappush(pq, (new_cost_to_v, v_candidate))
                
                # "Remove" `v_candidate` from `st_R` so it's not chosen again for this range query.
                # It means this particular (W_v, v_candidate) pair has been considered.
                # Store old value for reverting. `intervals[v_candidate][1]` is R_v_candidate.
                changes_to_revert_R.append((intervals[v_candidate][1], st_R.query(intervals[v_candidate][1], intervals[v_candidate][1], [False]*(N+1))))
                st_R.update(intervals[v_candidate][1], (INF, -1)) 

            # Find neighbors `v` where `L_v > R_u` (v's interval begins after u's ends)
            # Query `st_L` for minimum `W_v` in range `[R_u + 1, MAX_COORD_VALUE]`.
            while True:
                min_w_v, v_candidate = st_L.query(R_u + 1, MAX_COORD_VALUE, visited)
                
                if min_w_v == INF: # No more valid candidates in this range
                    break
                
                new_cost_to_v = current_path_weight - W[u] + min_w_v
                if new_cost_to_v < dist[v_candidate]:
                    dist[v_candidate] = new_cost_to_v
                    heapq.heappush(pq, (new_cost_to_v, v_candidate))
                
                # "Remove" `v_candidate` from `st_L`.
                # Store old value for reverting. `intervals[v_candidate][0]` is L_v_candidate.
                changes_to_revert_L.append((intervals[v_candidate][0], st_L.query(intervals[v_candidate][0], intervals[v_candidate][0], [False]*(N+1))))
                st_L.update(intervals[v_candidate][0], (INF, -1)) 
        
        # If PQ exhausted and target not reached, no path exists.
        # Revert all changes made to segment trees before returning -1.
        for coord, val in changes_to_revert_L:
            st_L.update(coord, val)
        for coord, val in changes_to_revert_R:
            st_R.update(coord, val)
        return -1

    # Process all queries.
    results = []
    for s, t in queries:
        results.append(dijkstra(s, t))
    
    # Print results to stdout.
    for res in results:
        sys.stdout.write(str(res) + "
")

# Call the main solve function.
solve()
# YOUR CODE HERE
import sys
import heapq

# Setting higher recursion depth might be needed for deep trees if using recursive DFS
# sys.setrecursionlimit(400010) # Example value -> Using iterative DFS instead

# Use iterative approach for DFS to avoid stack limits entirely.

def solve():
    N = int(sys.stdin.readline())
    
    # Handle the edge case N=1 separately. The problem states N >= 2,
    # but good practice to handle it if possible. If N=1, the only vertex is 1.
    # Aoki must choose K=1 vertex, which is vertex 1. Takahashi's walk starts and ends at 1,
    # must visit vertex 1. A walk of length 0 (staying at vertex 1) satisfies this. Score 0.
    if N == 1:
        print(0)
        return

    adj = [[] for _ in range(N)]
    for i in range(N - 1):
        # Read edge information: vertices u, v and length l
        u, v, l = map(int, sys.stdin.readline().split())
        # Adjust to 0-based indexing
        u -= 1
        v -= 1
        # Add edge to adjacency list for both vertices
        adj[u].append((v, l))
        adj[v].append((u, l))

    # Arrays to store tree properties computed by DFS
    parent = [-1] * N  # parent[i] stores the parent of node i in DFS tree rooted at 0
    depth = [-1] * N   # depth[i] stores the depth of node i (root depth is 0)
    dist = [-1] * N    # dist[i] stores the distance from root (node 0) to node i

    # Iterative DFS for tree traversal and precomputation
    # Stack stores tuples: (node, parent_idx, current_depth, current_distance)
    # Initial state: start DFS from node 0 (vertex 1 in problem statement)
    stack = [(0, -1, 0, 0)] 
    
    # Keep track of visited nodes to correctly assign parent/depth/dist in iterative DFS
    visited_dfs = {} # Using dictionary to track visited state
    
    # To manage the iterative DFS state, track how many children/neighbors processed for each node
    processed_neighbors = [0] * N 

    while stack:
        curr, p, d, di = stack[-1]
        
        # First time visiting the node `curr`
        if curr not in visited_dfs:
            visited_dfs[curr] = True # Mark as visited
            # Assign computed properties
            parent[curr] = p
            depth[curr] = d
            dist[curr] = di
        
        # Try to visit neighbors/children
        found_new_neighbor = False
        # Iterate through neighbors using the processed_neighbors counter
        while processed_neighbors[curr] < len(adj[curr]):
            neighbor, length = adj[curr][processed_neighbors[curr]]
            processed_neighbors[curr] += 1
            # Check parent carefully: 'p' from stack tuple is the node that pushed 'curr' onto stack.
            # Avoid going back up the edge we came down from.
            if neighbor != p: 
                 # If neighbor hasn't been visited yet, push it onto the stack
                 if neighbor not in visited_dfs:
                     stack.append((neighbor, curr, d + 1, di + length))
                     found_new_neighbor = True
                     break # DFS proceeds down this new path; break inner loop to process new top of stack
        
        # If no unvisited neighbor was found, it means all children subtrees are explored.
        # We are done with this node, so backtrack.
        if not found_new_neighbor: 
            stack.pop() # Pop current node from stack


    # --- Main Greedy Algorithm Part ---
    
    # `active[i]` is True if vertex i is on a path from some chosen vertex to the root (node 0)
    active = [False] * N
    active[0] = True # vertex 1 (index 0) is initially active base for paths
    
    # `NAA[i]` stores the index of the Nearest Active Ancestor for node i
    NAA = [0] * N # Initially, the root (node 0) is the NAA for all other nodes
    
    # Max heap to efficiently find the vertex with the maximum marginal benefit
    max_heap = []
    # `benefits` map stores current valid benefit value for nodes. Key: node_idx, Value: benefit
    # Used to check if heap entries are outdated
    benefits = {} 

    # Initialize benefits for all nodes except the root (node 0)
    for i in range(1, N):
        benefit = dist[i] # Initial benefit is distance from root
        # Only consider nodes with positive path length to root, which contribute benefit
        if benefit > 0 : 
           benefits[i] = benefit
           # Use negative benefit for max heap behavior with heapq (min heap)
           heapq.heappush(max_heap, (-benefit, i)) 

    # `current_total_covered_length` tracks the sum of edge lengths in the minimal subtree covering chosen nodes + root
    current_total_covered_length = 0
    results = [] # Stores the final score for each K

    # `chosen_vertices` set tracks vertices explicitly chosen by Aoki (plus vertex 0 implicitly)
    chosen_vertices = {0} 

    # Iterate for K = 1 to N
    for k in range(1, N + 1):
        
        best_benefit = 0 # Default benefit if heap empty or only invalid items remain
        best_v = -1 # Index of the best vertex to choose

        # Find the vertex with max benefit among valid candidates
        while max_heap:
            # Peek top element (max benefit candidate)
            neg_benefit, v_idx = max_heap[0] 
            
            # Check if the heap entry is outdated or invalid
            is_outdated_or_invalid = False
            if v_idx not in benefits: # Benefit entry removed from map?
                 is_outdated_or_invalid = True
            # Benefit became non-positive? (Node effectively covered)
            elif benefits[v_idx] <= 0: 
                 is_outdated_or_invalid = True
            # Benefit value in map differs from heap entry?
            elif -neg_benefit != benefits[v_idx]: 
                 is_outdated_or_invalid = True
            
            if is_outdated_or_invalid:
                 heapq.heappop(max_heap) # Remove outdated/invalid entry
                 # If benefit became non-positive, clean up map potentially
                 if v_idx in benefits and benefits[v_idx] <= 0:
                      del benefits[v_idx]
                 continue 

            # Check if vertex already chosen
            if v_idx in chosen_vertices:
                 heapq.heappop(max_heap) # Remove already chosen vertex
                 if v_idx in benefits: # Clean up map after selection
                    del benefits[v_idx]
                 continue

            # Valid vertex found
            best_benefit = -neg_benefit
            best_v = v_idx
            heapq.heappop(max_heap) # Remove the chosen vertex from heap
            if v_idx in benefits: # Clean up map after selection
                del benefits[v_idx]
            break # Exit loop after finding the best vertex
        
        # Handle case where no valid vertex with positive benefit is found
        if best_v == -1 :
             # Score doesn't increase further. Append the last calculated score.
             if results:
                 results.append(results[-1])
             else:
                 # If K=1 and no valid vertex, score is 0.
                 results.append(0) 
             continue

        # Select vertex `best_v`
        chosen_vertices.add(best_v)
        # Update total covered length by adding the marginal benefit
        current_total_covered_length += best_benefit
        # Calculate and store the score for current K. Score is 2 * total covered length.
        results.append(current_total_covered_length * 2)

        # --- Update Phase ---
        # Activate the path from the chosen vertex `best_v` up towards the root,
        # stopping at its Nearest Active Ancestor `NAA[best_v]`.
        target_ancestor = NAA[best_v]
        
        v_on_path = best_v
        # Collect nodes on the path segment to be activated
        # We process updates after identifying the whole path segment
        path_nodes_to_process_updates = [] 

        # Traverse up from best_v towards target_ancestor
        while v_on_path != target_ancestor:
             # Safety check: If node is already active, stop. Shouldn't happen with correct NAA logic.
             if active[v_on_path]: 
                  break 
             
             active[v_on_path] = True # Mark node as active
             path_nodes_to_process_updates.append(v_on_path) # Store node for update processing step
             
             # Stop if root is reached
             if v_on_path == 0: 
                  break
             # Check parent exists before accessing parent array
             if parent[v_on_path] == -1: 
                  break
             # Move up to the parent node
             v_on_path = parent[v_on_path]

        # Process updates triggered by activating nodes on the path segment
        for v_node in path_nodes_to_process_updates:
             
            # Use BFS for update propagation into subtrees rooted at children
            # This avoids deep recursion.
            queue_update = [] # Initialize queue for BFS

            # Iterate through neighbors of `v_node`
            for neighbor, length in adj[v_node]:
                # Consider only children nodes (not the parent)
                if neighbor != parent[v_node]: 
                    # Explore subtree rooted at this child if it's not already active
                    if not active[neighbor]:
                         queue_update.append(neighbor)

            # Perform BFS starting from children
            head_idx = 0
            while head_idx < len(queue_update):
                 curr_update_node = queue_update[head_idx]
                 head_idx += 1
                 
                 # If node became active through another path concurrently, skip
                 if active[curr_update_node]: 
                     continue

                 # --- Key Update Condition ---
                 # Check if the newly activated node `v_node` is closer (lower depth) 
                 # than the current Nearest Active Ancestor `NAA[curr_update_node]`
                 # Depth check: depth[v_node] > depth[NAA[curr_update_node]]
                 if depth[v_node] > depth[NAA[curr_update_node]]:
                    # Update NAA to the closer active node `v_node`
                    NAA[curr_update_node] = v_node
                    # Calculate new marginal benefit: distance to the new NAA
                    new_benefit = dist[curr_update_node] - dist[v_node]
                    
                    # Update benefits map and heap only if benefit is positive and changed
                    if new_benefit > 0:
                        # Check if benefit value actually changed before updating
                        if curr_update_node not in benefits or benefits[curr_update_node] != new_benefit:
                            benefits[curr_update_node] = new_benefit
                            # Push updated benefit (negated) to max heap
                            heapq.heappush(max_heap, (-new_benefit, curr_update_node))
                    else: # Benefit became zero or negative
                        # Remove from benefits map if exists
                        if curr_update_node in benefits:
                            # Setting benefit to 0 ensures outdated heap entries are ignored later
                            benefits[curr_update_node] = 0 
                            # Explicit deletion from map might be cleaner: del benefits[curr_update_node]

                    # Continue BFS exploration down to children of `curr_update_node`
                    for neighbor, length in adj[curr_update_node]:
                        # Ensure moving down to children, not back to parent
                        if neighbor != parent[curr_update_node]: 
                            # Explore only if child is not active
                            if not active[neighbor]: 
                                queue_update.append(neighbor)
                 # else: Update condition not met (v_node is not closer than current NAA).
                 # This branch of subtree doesn't need further updates originating from v_node. Stop descent this path.


    # Print the computed scores for K=1 to N
    for score in results:
        print(score)
        
solve()
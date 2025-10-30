import sys
import collections

def main():
    N, M = map(int, sys.stdin.readline().split())

    adj = [[] for _ in range(N + 1)]
    # parent_of[i] stores the parent of node i. parent_of[1] = 0 (root has no "problem-defined" parent).
    parent_of = [0] * (N + 1) 

    # Constraints: 2 <= N. So N-1 >= 1.
    # The parents p_2 ... p_N are N-1 integers.
    # parents_input is 0-indexed. parents_input[k] is the parent of node k+2.
    if N > 1: # N >= 2 always true from constraints, so this if is not strictly needed
        parents_input = list(map(int, sys.stdin.readline().split()))
        for i in range(N - 1): # Iterate N-1 times, for p_2 through p_N
            child_node = i + 2 # child_node goes from 2 to N
            parent_node = parents_input[i] # parents_input[0] is p_2, ..., parents_input[N-2] is p_N
            
            adj[parent_node].append(child_node)
            parent_of[child_node] = parent_node

    # depth[i] stores the depth of node i. Root (node 1) is at depth 0.
    depth = [-1] * (N + 1) 
    
    # BFS queue
    q = collections.deque()
    # bfs_nodes_order stores nodes in the order they are processed by BFS.
    # This order ensures parents are processed before their children.
    bfs_nodes_order = []

    # Initialize BFS starting from root (node 1)
    # Constraints: N >= 2, so node 1 always exists.
    q.append(1)
    depth[1] = 0
    
    while q:
        u = q.popleft()
        bfs_nodes_order.append(u)
        
        for v in adj[u]:
            # In a tree, children are discovered for the first time via their parent.
            if depth[v] == -1: 
                depth[v] = depth[u] + 1
                q.append(v)

    # For each node x, max_reach_from_node[x] stores the maximum depth
    # (depth[x] + y_val) that an insurance bought directly by x can cover.
    # Initialized with -1 (representing negative infinity, as depths are non-negative).
    max_reach_from_node = [-1] * (N + 1) 

    for _ in range(M):
        x, y_val = map(int, sys.stdin.readline().split())
        # An insurance bought by x covers descendants up to y_val generations.
        # This means it covers nodes u in subtree of x where depth[u] <= depth[x] + y_val.
        # So, this insurance provides coverage up to absolute depth: depth[x] + y_val.
        # depth[x] could be -1 if x is not part of the tree rooted at 1, but problem constraints imply a connected tree.
        if depth[x] != -1 : # Should always be true for valid x
            coverage_depth_limit = depth[x] + y_val
            # If multiple insurances are bought by person x, take the one that covers deepest.
            if coverage_depth_limit > max_reach_from_node[x]:
                max_reach_from_node[x] = coverage_depth_limit
            
    # effective_max_reach[u] will store the maximum coverage depth limit
    # applicable to node u, considering insurances bought by u OR any of its ancestors.
    effective_max_reach = [-1] * (N + 1)
    
    count_covered_people = 0

    # Iterate through nodes in BFS order (parents are processed before their children).
    for u in bfs_nodes_order:
        # Determine the maximum coverage depth inherited from u's parent.
        p_u = parent_of[u] # Parent of u
        inherited_reach = -1 
        if p_u != 0: # If u is not the root (root has no parent to inherit from)
            inherited_reach = effective_max_reach[p_u]
        
        # Determine the maximum coverage depth from an insurance bought by u itself.
        own_reach = max_reach_from_node[u]
        
        # The overall maximum coverage depth for node u is the max of what's inherited
        # and what u provides itself.
        current_node_overall_reach = max(inherited_reach, own_reach)
        effective_max_reach[u] = current_node_overall_reach
        
        # A person u is covered if their depth is less than or equal to the
        # maximum reach applicable to them.
        # depth[u] will always be >= 0 for nodes in bfs_nodes_order.
        if effective_max_reach[u] != -1 and depth[u] <= effective_max_reach[u]:
            count_covered_people += 1
            
    print(count_covered_people)

if __name__ == '__main__':
    main()
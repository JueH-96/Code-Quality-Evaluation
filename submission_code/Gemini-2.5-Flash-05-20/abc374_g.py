import sys
from collections import defaultdict, deque

def solve():
    N = int(sys.stdin.readline())
    
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    # Store adjacency lists for WCC traversal
    # Using a set for adj values to handle duplicate products gracefully, though problem states distinct S_i
    # For WCC, we need to traverse in both directions, so we build both adj and rev_adj
    adj = defaultdict(set)
    rev_adj = defaultdict(set)
    
    # Keep track of all characters that actually participate in edges
    all_active_chars = set()

    for _ in range(N):
        S = sys.stdin.readline().strip()
        u, v = S[0], S[1]
        
        out_degree[u] += 1
        in_degree[v] += 1
        
        adj[u].add(v)
        rev_adj[v].add(u)
        
        all_active_chars.add(u)
        all_active_chars.add(v)

    min_strings = 0
    
    # Track visited nodes for WCC calculation
    visited_for_wcc = set()

    # Calculate total "excess starts" from non-Eulerian components.
    # This is also the number of paths that cannot be continued by a previous edge.
    # The sum of (out_degree - in_degree) for nodes within a component where it's positive.
    
    # Use BFS/DFS to find all weakly connected components (WCCs)
    # A WCC is a connected component in the underlying undirected graph.
    # To find WCCs, we traverse both forward (adj) and reverse (rev_adj) edges.
    
    for char_code in range(ord('A'), ord('Z') + 1):
        start_node = chr(char_code)
        
        # Check if the node is part of the graph (has any associated edges)
        # and has not been visited yet by a previous WCC traversal
        if (start_node in all_active_chars) and (start_node not in visited_for_wcc):
            
            # Found a new WCC
            current_wcc_nodes = set()
            q = deque()
            
            q.append(start_node)
            visited_for_wcc.add(start_node)
            current_wcc_nodes.add(start_node)
            
            while q:
                u = q.popleft()
                
                # Traverse outgoing edges
                for v in adj[u]:
                    if v not in visited_for_wcc:
                        visited_for_wcc.add(v)
                        current_wcc_nodes.add(v)
                        q.append(v)
                
                # Traverse incoming edges (for reverse adjacency)
                for v in rev_adj[u]:
                    if v not in visited_for_wcc:
                        visited_for_wcc.add(v)
                        current_wcc_nodes.add(v)
                        q.append(v)
            
            # Calculate excess_starts for the current WCC
            wcc_excess_starts = 0
            for node in current_wcc_nodes:
                wcc_excess_starts += max(0, out_degree[node] - in_degree[node])
            
            # Apply the rule based on WCC_excess_starts
            if wcc_excess_starts == 0:
                # If all nodes in this WCC have in_degree == out_degree, it's an Eulerian component.
                # It requires exactly 1 string to cover all its edges.
                min_strings += 1
            else:
                # If there are degree imbalances, this component contributes
                # 'wcc_excess_starts' number of distinct paths.
                min_strings += wcc_excess_starts
                
    sys.stdout.write(str(min_strings) + "
")

solve()
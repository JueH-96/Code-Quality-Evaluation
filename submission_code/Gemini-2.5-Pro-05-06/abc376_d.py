import sys
import collections

def main():
    # Use a local binding for sys.stdin.readline for potentially faster I/O
    _input = sys.stdin.readline
    
    N, M = map(int, _input().split())
    
    # Adjacency lists for forward and reverse edges
    # Vertices are 0-indexed internally (original 1..N maps to 0..N-1)
    adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)] # rev_adj[v] stores predecessors of v
    
    for _ in range(M):
        u, v = map(int, _input().split())
        u_idx = u - 1 # Convert to 0-indexed
        v_idx = v - 1 # Convert to 0-indexed
        
        adj[u_idx].append(v_idx)
        rev_adj[v_idx].append(u_idx) # u_idx is a predecessor of v_idx

    # The problem asks for cycles involving vertex 1.
    # In 0-indexed system, this is source_node_idx = 0.
    source_node_idx = 0 
    
    # dist[i] will store the shortest path distance from source_node_idx to i
    dist = [-1] * N 
    
    # Queue for BFS
    q = collections.deque()

    # Initialize BFS from source_node_idx
    dist[source_node_idx] = 0
    q.append(source_node_idx)
    
    # BFS main loop
    while q:
        curr_u = q.popleft()
        
        for neighbor_v in adj[curr_u]:
            if dist[neighbor_v] == -1: # If neighbor_v has not been visited
                dist[neighbor_v] = dist[curr_u] + 1
                q.append(neighbor_v)

    min_cycle_len = float('inf')

    # Iterate through all predecessors of source_node_idx.
    # An edge (u_pred_idx -> source_node_idx) can complete a cycle
    # if u_pred_idx is reachable from source_node_idx.
    # The cycle would be: 
    # source_node_idx -> ... (path of length dist[u_pred_idx]) ... -> u_pred_idx -> source_node_idx
    # Total length of this cycle = dist[u_pred_idx] + 1.
    for u_pred_idx in rev_adj[source_node_idx]:
        # u_pred_idx is a node that has an edge TO source_node_idx
        if dist[u_pred_idx] != -1: # Check if u_pred_idx is reachable from source_node_idx
            cycle_len = dist[u_pred_idx] + 1
            min_cycle_len = min(min_cycle_len, cycle_len)
                
    # Output result
    if min_cycle_len == float('inf'):
        sys.stdout.write("-1
")
    else:
        sys.stdout.write(str(min_cycle_len) + "
")

if __name__ == '__main__':
    main()
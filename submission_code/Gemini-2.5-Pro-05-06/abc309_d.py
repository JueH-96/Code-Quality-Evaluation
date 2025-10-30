import collections
import sys

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    N_total = N1 + N2

    adj = [[] for _ in range(N_total)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed vertices
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    # BFS function to find the maximum distance from a start node to any other node
    # within a specified component of the graph.
    def bfs_component_max_dist(start_node_0_indexed, is_node_in_component_func, current_adj, num_nodes):
        dist = [-1] * num_nodes
        dist[start_node_0_indexed] = 0
        
        q = collections.deque()
        q.append(start_node_0_indexed)
        
        while q:
            u = q.popleft()
            
            for v_neighbor in current_adj[u]:
                # The problem guarantees V1 and V2 are disconnected. This implies that if u
                # is in a component (e.g., V1), all its neighbors v_neighbor must also be
                # in the same component. Thus, is_node_in_component_func(v_neighbor)
                # would effectively be true if not for potential isolated nodes (which are
                # ruled out by connectivity guarantees within V1 and V2).
                # This check ensures we only consider paths within the component.
                if is_node_in_component_func(v_neighbor) and dist[v_neighbor] == -1:
                    dist[v_neighbor] = dist[u] + 1
                    q.append(v_neighbor)
        
        max_dist_found = 0
        # Iterate through all nodes to find the maximum distance to a node within the component
        for i in range(num_nodes):
            if is_node_in_component_func(i):
                # All nodes in the component are guaranteed to be reachable.
                # If dist[i] == -1, it would contradict problem statement's connectivity.
                # assert dist[i] != -1 
                max_dist_found = max(max_dist_found, dist[i])
        
        return max_dist_found

    # Calculate max_d1: Max shortest distance from vertex 1 to any vertex in V1
    # V1: nodes 0 to N1-1. Start from node 0 (original vertex 1).
    is_in_V1 = lambda i: 0 <= i < N1
    max_d1 = bfs_component_max_dist(0, is_in_V1, adj, N_total)

    # Calculate max_d2: Max shortest distance from vertex N1+N2 to any vertex in V2
    # V2: nodes N1 to N_total-1. Start from node N_total-1 (original vertex N1+N2).
    start_node_V2 = N_total - 1
    is_in_V2 = lambda i: N1 <= i < N_total
    max_d2 = bfs_component_max_dist(start_node_V2, is_in_V2, adj, N_total)
    
    print(max_d1 + max_d2 + 1)

if __name__ == '__main__':
    main()
import sys
import collections

def solve():
    N = int(sys.stdin.readline())
    
    adj = [[] for _ in range(N)]
    # For N=1 (not allowed by constraints N>=2, but if it were):
    # this loop range is 0, so it doesn't run. adj[0] would be [].
    for _ in range(N - 1): 
        u, v = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    # Vertex 1 is target_node_idx = 0
    target_node_idx = 0 

    max_subtree_size = 0
    
    # If target_node_idx has no children (adj[target_node_idx] is empty),
    # this loop won't run. max_subtree_size will remain 0.
    # This occurs if N=1. Result N - 0 = N. For N=1, this is 1.
    # For N >= 2, target_node_idx must have at least one child as it's a tree.
    for child_node in adj[target_node_idx]:
        # Calculate component size using BFS
        q = collections.deque()
        
        # visited_bfs stores nodes visited in current BFS or forbidden (target_node_idx)
        # target_node_idx is "forbidden" to ensure we only count nodes in the child's component.
        visited_bfs = {target_node_idx} 
        
        q.append(child_node)
        visited_bfs.add(child_node)
        
        current_component_size = 0
        while q:
            u = q.popleft()
            current_component_size += 1
            
            for v_neighbor in adj[u]:
                if v_neighbor not in visited_bfs:
                    visited_bfs.add(v_neighbor)
                    q.append(v_neighbor)
        
        if current_component_size > max_subtree_size:
            max_subtree_size = current_component_size
            
    print(N - max_subtree_size)

if __name__ == '__main__':
    solve()
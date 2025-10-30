import sys
from collections import deque

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    A_seq = list(map(int, sys.stdin.readline().split()))
    B_seq = list(map(int, sys.stdin.readline().split()))

    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u, v = A_seq[i], B_seq[i]
        
        # If u == v, it's a self-loop. X_u must be different from X_u, which is impossible.
        # A self-loop is an odd cycle of length 1.
        # The general BFS check below would also detect this, but an early exit is clear and efficient.
        if u == v:
            print("No")
            return
        
        adj[u].append(v)
        adj[v].append(u)

    # colors array: -1 means uncolored, 0 means color 0, 1 means color 1.
    # Vertices are 1-indexed, so array size is N+1.
    colors = [-1] * (N + 1)  
    
    is_bipartite_globally = True 

    # Iterate through all vertices to handle disconnected graphs
    for i in range(1, N + 1):
        if colors[i] == -1:  # If vertex i is uncolored, it's part of a new component
            colors[i] = 0    # Start coloring this component: vertex i gets color 0
            queue = deque([i])
            
            current_component_is_bipartite = True # Flag for the current component
            
            # BFS for the current component
            while queue:
                u_node = queue.popleft()
                for v_node in adj[u_node]:
                    if colors[v_node] == -1: # If neighbor v_node is uncolored
                        colors[v_node] = 1 - colors[u_node] # Assign opposite color
                        queue.append(v_node)
                    elif colors[v_node] == colors[u_node]: # If neighbor has same color
                        current_component_is_bipartite = False
                        break # Conflict found, this component is not bipartite
                
                if not current_component_is_bipartite:
                    break # Exit BFS for this component (while queue loop)
            
            if not current_component_is_bipartite:
                is_bipartite_globally = False
                break # Entire graph is not bipartite, no need to check other components
                
    if is_bipartite_globally:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()
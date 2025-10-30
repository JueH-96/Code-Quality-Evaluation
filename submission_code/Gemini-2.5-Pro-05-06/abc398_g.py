import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    component_processing_visited = [False] * N
    g_values = [] 
    first_component_actual_diff = None

    for i in range(N):
        if not component_processing_visited[i]:
            s0_size = 0 
            s1_size = 0 
            
            q = [] 
            q.append((i, 0)) # (vertex, color)
            component_processing_visited[i] = True
            
            bfs_colors = {} 
            bfs_colors[i] = 0
            
            head = 0
            component_nodes_count = 0
            while head < len(q):
                curr, color = q[head]
                head += 1
                component_nodes_count +=1
                
                if color == 0:
                    s0_size += 1
                else:
                    s1_size += 1
                
                for neighbor in adj[curr]:
                    if neighbor not in bfs_colors: # Not yet visited in this component's BFS
                        if component_processing_visited[neighbor]:
                             # This implies neighbor was visited as part of another component's BFS run.
                             # This should not happen if BFS explores full components.
                             # It means adj[curr] has an edge to an already processed component.
                             # This is fine, indicates graph structure is complex, but BFS handles it.
                             # The check `if not component_processing_visited[neighbor]` for adding to queue is key.
                             pass # Should not be an issue.
                        
                        component_processing_visited[neighbor] = True
                        bfs_colors[neighbor] = 1 - color
                        q.append((neighbor, 1 - color))
            
            if first_component_actual_diff is None:
                 first_component_actual_diff = s0_size - s1_size
            else: 
                 g_values.append(abs(s0_size - s1_size))

    if N == 0:
        print("Takahashi")
        return

    if first_component_actual_diff is None: # N > 0, but no components found (e.g. M=0, all isolated)
        first_component_actual_diff = 1 # From the first isolated vertex (1,0) partition
        for _ in range(N - 1): 
            g_values.append(1) # Other N-1 isolated vertices
    
    g_values.sort(reverse=True)
    current_D = first_component_actual_diff

    for j in range(len(g_values)):
        g_val = g_values[j]
        is_aoki_turn = (j % 2 == 0)

        d_plus = current_D + g_val
        d_minus = current_D - g_val
        abs_plus = abs(d_plus)
        abs_minus = abs(d_minus)

        if is_aoki_turn: 
            if abs_plus < abs_minus:
                current_D = d_plus
            elif abs_minus < abs_plus:
                current_D = d_minus
            else: 
                current_D = min(d_plus, d_minus)
        else: 
            if abs_plus > abs_minus:
                current_D = d_plus
            elif abs_minus > abs_plus:
                current_D = d_minus
            else: 
                current_D = max(d_plus, d_minus)
    
    max_edges = (N*N - current_D*current_D) // 4
    num_moves = max_edges - M

    if num_moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()
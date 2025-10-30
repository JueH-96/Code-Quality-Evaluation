import sys
from collections import deque

def solve():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * (N + 1)
    components = [] # List of (s0, s1) tuples

    for i in range(1, N + 1):
        if not visited[i]:
            q = deque([i])
            visited[i] = True
            
            color = [-1] * (N + 1)
            color[i] = 0
            partition_sizes = [0, 0]
            partition_sizes[0] += 1
            
            component_node_list = [i]
            head = 0

            while head < len(component_node_list):
                u = component_node_list[head]
                head += 1

                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        color[v] = 1 - color[u]
                        partition_sizes[color[v]] += 1
                        component_node_list.append(v)
                    # No need to check for odd cycle, problem guarantees initial graph is bipartite
            
            components.append(tuple(partition_sizes))
    
    k = len(components)
    isolated_count = 0
    non_isolated_components = []
    
    for s0, s1 in components:
        if s0 + s1 == 1:
            isolated_count += 1
        else:
            non_isolated_components.append((s0, s1))
            
    k_2 = len(non_isolated_components)

    if k == 1:
        s0, s1 = components[0]
        total_moves = s0 * s1 - M
        if total_moves % 2 == 1:
            print("Aoki")
        else:
            print("Takahashi")
    else: # k > 1
        if k_2 == 0: # All components are isolated vertices, M must be 0
            # k = N, M = 0
            # The game ends when the graph is a complete bipartite graph, partitions S0, S1 summing to N.
            # Any partition (S0, S1) with S0+S1=N can be formed from N isolated vertices.
            # To maximize moves (and win), the player forms partitions N/2, N-N/2.
            total_moves = (N // 2) * (N - N // 2) # floor(N/2) * ceil(N/2)
            if total_moves % 2 == 1:
                print("Aoki")
            else:
                print("Takahashi")
        else: # k_2 >= 1
            # Check if there is any non-trivial component with odd size
            has_odd_non_trivial_component = False
            for s0, s1 in non_isolated_components:
                if (s0 + s1) % 2 == 1:
                    has_odd_non_trivial_component = True
                    break

            if isolated_count >= 1 or has_odd_non_trivial_component:
                # If there are isolated vertices or non-trivial components of odd size,
                # players can influence the final partition configuration (S0, S1).
                # This flexibility allows the first player (Aoki) to win.
                print("Aoki")
            else: # k_1 == 0 and all non-trivial components have even size
                # N must be even (sum of even component sizes)
                # All non-trivial components have s0 = s1 = n/2.
                # The final combined partition (S0, S1) must be (N/2, N/2).
                # Total moves = (N/2) * (N/2) - M
                total_moves = (N // 2) * (N // 2) - M
                if total_moves % 2 == 1:
                    print("Aoki")
                else:
                    print("Takahashi")
                

solve()
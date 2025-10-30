import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    
    # Read edges and build adjacency list and degrees
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1
            
    ans = 0
    # path_checked_pairs stores frozensets of (u, v) to count each valid pair exactly once
    path_checked_pairs = set() 
    
    # Iterate through all vertices. Only start BFS from potential 'x' vertices (degree 2).
    for i in range(1, N + 1):
        if degrees[i] == 2:
            # BFS state: (current_node, start_node_of_this_bfs, path_length)
            q = deque([(i, i, 0)])
            # visited_this_bfs tracks nodes visited in the current BFS traversal
            # to ensure simple paths and prevent cycles within the BFS exploration.
            visited_this_bfs = {i} 
            
            while q:
                u, start_node, length = q.popleft()
                
                # Condition for 'y': u is a D2 node and is not the start_node,
                # and the path length is at least 2 (ensures non-adjacency and at least one intermediate node).
                if u != start_node and degrees[u] == 2 and length >= 2:
                    pair = frozenset({start_node, u})
                    if pair not in path_checked_pairs:
                        ans += 1
                        path_checked_pairs.add(pair)
                
                # Logic to determine if we can extend the path from 'u':
                # We can extend if 'u' is the start_node (which is D2), or if 'u' is a D3 node.
                # If 'u' is a D2 node but not the start_node, it means it's a potential 'y' and we stop extending from it.
                if degrees[u] == 2 and u != start_node:
                    continue # Do not explore neighbors from a non-start D2 node in the path.
                
                # Explore neighbors of 'u'
                for v in adj[u]:
                    if v not in visited_this_bfs:
                        # Valid nodes to extend the path are D3 nodes or another D2 node (which would be 'y').
                        if degrees[v] == 3: # Can traverse through D3 nodes
                            visited_this_bfs.add(v)
                            q.append((v, start_node, length + 1))
                        elif degrees[v] == 2: # Can terminate at a D2 node (potential 'y')
                            visited_this_bfs.add(v)
                            q.append((v, start_node, length + 1))
                        # If degrees[v] is 1 or >3, 'v' cannot be on the path, so skip.
                            
    print(ans)

solve()
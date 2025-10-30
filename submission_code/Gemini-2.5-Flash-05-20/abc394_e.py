import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    
    C_matrix = []
    for _ in range(N):
        C_matrix.append(sys.stdin.readline().strip())
    
    # Adjacency lists for graph and reverse graph
    # adj[u] will store (v, char_label) for edges u -> v
    # rev_adj[u] will store (v, char_label) for edges v -> u
    adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            char_label = C_matrix[i][j]
            if char_label != '-':
                adj[i].append((j, char_label))
                rev_adj[j].append((i, char_label))
                
    # dist[u][v] will store the shortest length of a palindromic path from u to v
    # Initialize with infinity
    dist = [[float('inf')] * N for _ in range(N)]
    
    # BFS Queue: stores (u, v, current_path_length)
    q = deque()
    
    # Base cases for palindromic paths:
    # 1. Path of length 0 (empty string is a palindrome)
    for i in range(N):
        dist[i][i] = 0
        q.append((i, i, 0))
        
    # 2. Path of length 1 (single character is a palindrome)
    for i in range(N):
        for j in range(N):
            if C_matrix[i][j] != '-':
                # A path of length 1 from i to j has label C_matrix[i][j].
                # This single character is always a palindrome.
                # If dist[i][j] is currently infinity (meaning no path found yet)
                # or is greater than 1 (meaning a longer path was found or default value),
                # update it to 1 and add to queue.
                # Note: if i==j and C_matrix[i][i] is a char, dist[i][i] is 0.
                # 0 is not > 1, so it won't be updated, which is correct (length 0 is shorter).
                if dist[i][j] > 1: 
                    dist[i][j] = 1
                    q.append((i, j, 1))
                    
    # BFS loop
    while q:
        u, v, d = q.popleft()
        
        # If we found a shorter path to (u, v) already, skip
        if d > dist[u][v]:
            continue
            
        # Try to extend the current palindromic path (u -> ... -> v)
        # by adding one edge before u and one edge after v with matching labels.
        # This forms a new palindromic path: prev_u -> u -> ... -> v -> next_v
        
        for prev_u, char_prev in rev_adj[u]: # Iterate through edges leading into 'u'
            for next_v, char_next in adj[v]: # Iterate through edges leading out of 'v'
                if char_prev == char_next: # Check if the labels match
                    new_d = d + 2 # The new path length is current_length + 2
                    if new_d < dist[prev_u][next_v]: # If this is a shorter path
                        dist[prev_u][next_v] = new_d
                        q.append((prev_u, next_v, new_d))
                        
    # Format and print output
    result_lines = []
    for i in range(N):
        row_output = []
        for j in range(N):
            if dist[i][j] == float('inf'):
                row_output.append("-1")
            else:
                row_output.append(str(dist[i][j]))
        result_lines.append(" ".join(row_output))
        
    sys.stdout.write("
".join(result_lines) + "
")

solve()
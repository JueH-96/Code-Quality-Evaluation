import sys
import collections

def solve():
    # Read N1, N2, M
    N1, N2, M = map(int, sys.stdin.readline().split())
    N_total = N1 + N2

    # Initialize adjacency list for the graph (1-indexed)
    adj = [[] for _ in range(N_total + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # --- BFS from vertex 1 ---
    # dist1 stores shortest distances from vertex 1. Initialize with -1 (unvisited).
    dist1 = [-1] * (N_total + 1)
    q1 = collections.deque()

    # Start BFS from vertex 1
    dist1[1] = 0
    q1.append(1)
    max_d1_val = 0 # Stores the maximum distance from vertex 1 to any node in [1, N1]

    while q1:
        curr = q1.popleft()
        
        # If the current vertex is in the first component [1, N1], update max_d1_val
        if 1 <= curr <= N1:
            max_d1_val = max(max_d1_val, dist1[curr])
        
        # Explore neighbors
        for neighbor in adj[curr]:
            if dist1[neighbor] == -1: # If neighbor not visited
                dist1[neighbor] = dist1[curr] + 1
                q1.append(neighbor)
    
    # --- BFS from vertex N_total (N1+N2) ---
    # distN stores shortest distances from vertex N_total. Initialize with -1.
    distN = [-1] * (N_total + 1)
    qN = collections.deque()

    # Start BFS from vertex N_total
    distN[N_total] = 0
    qN.append(N_total)
    max_dN_val = 0 # Stores the maximum distance from vertex N_total to any node in [N1+1, N_total]

    while qN:
        curr = qN.popleft()
        
        # If the current vertex is in the second component [N1+1, N_total], update max_dN_val
        if N1 + 1 <= curr <= N_total:
            max_dN_val = max(max_dN_val, distN[curr])
        
        # Explore neighbors
        for neighbor in adj[curr]:
            if distN[neighbor] == -1: # If neighbor not visited
                distN[neighbor] = distN[curr] + 1
                qN.append(neighbor)

    # The maximum possible d is the sum of the maximum distances found in each component
    # plus 1 for the newly added edge (u, v).
    print(max_d1_val + max_dN_val + 1)

# Call the solve function to run the program
solve()
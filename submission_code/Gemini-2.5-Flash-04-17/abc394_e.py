import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    C = [sys.stdin.readline().strip() for _ in range(N)]

    # Build adjacency lists: adj_list[u][label_idx] = list of v
    # and reversed adjacency lists: rev_adj_list[v][label_idx] = list of u
    # where u and v are 0-based indices (0 to N-1)
    adj_list = [[[] for _ in range(26)] for _ in range(N)]
    rev_adj_list = [[[] for _ in range(26)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if C[i][j] != '-':
                label = C[i][j]
                label_idx = ord(label) - ord('a')
                
                # Edge from i to j with label C[i][j] (0-based indices)
                adj_list[i][label_idx].append(j) # From u=i with label, can go to v=j
                rev_adj_list[j][label_idx].append(i) # To v=j with label, can come from u=i


    # min_len[i][j] will store the shortest palindrome path length from i to j (0-based)
    # Initialize with -1 to represent infinity/unvisited
    min_len = [[-1] * N for _ in range(N)]

    # Queue for BFS storing pairs of vertices (u, v)
    Q = deque()

    # Base cases: Length 0 paths (from i to i)
    # The empty path is a palindrome.
    for i in range(N):
        min_len[i][i] = 0
        Q.append((i, i))

    # Base cases: Length 1 paths (direct edges i -> j)
    # Any single character string is a palindrome.
    # Add these after length 0 states to process by distance layer.
    for i in range(N):
        for j in range(N):
            if i != j and C[i][j] != '-':
                 # A direct edge i -> j provides a palindrome path of length 1.
                 # Only add if we haven't found a shorter path already (which is impossible at this stage for length 1)
                if min_len[i][j] == -1: # Should always be -1 for i != j initialized states
                    min_len[i][j] = 1
                    Q.append((i, j))

    # BFS
    # The queue stores pairs (u, v). The distance to (u, v) is min_len[u][v].
    # BFS explores states layer by layer, guaranteeing shortest paths.
    while Q:
        u, v = Q.popleft()
        d = min_len[u][v] # Current shortest distance found for (u, v)

        # Explore transitions from state (u, v)
        # If a palindrome path from u to v of length d exists,
        # we can potentially form a longer palindrome path from x to y of length d + 2
        # by adding edges x -> u and v -> y with the same label L.
        
        # Iterate through all possible edge labels L
        for label_idx in range(26):
            # Find all vertices x such that an edge x -> u exists with this label L
            # These are vertices in rev_adj_list[u][label_idx]
            prev_nodes = rev_adj_list[u][label_idx]
            
            # Find all vertices y such that an edge v -> y exists with this label L
            # These are vertices in adj_list[v][label_idx]
            next_nodes = adj_list[v][label_idx]

            # For every combination of such x and y
            for x in prev_nodes:
                for y in next_nodes:
                    # We found a palindrome path from x to y of length d + 2
                    # Check if this is a shorter path than currently known for (x, y)
                    if min_len[x][y] == -1:
                        min_len[x][y] = d + 2
                        Q.append((x, y))

    # Output the result
    for i in range(N):
        print(' '.join(map(str, min_len[i])))

solve()
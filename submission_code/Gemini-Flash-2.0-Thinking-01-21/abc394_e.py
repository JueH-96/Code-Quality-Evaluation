import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    C = [sys.stdin.readline().strip() for _ in range(N)]

    # dist[i][j] will store the shortest palindrome path length from vertex i to vertex j (0-indexed)
    # Initialize with a value representing infinity. A value larger than any possible shortest path length.
    # The maximum number of states is N*N. Each step in BFS adds 2 to the distance.
    # A loose upper bound on the maximum distance is roughly 1 + 2 * (N*N - 1).
    INF = 2 * N * N + 7 
    dist = [[INF for _ in range(N)] for _ in range(N)]

    # Queue for BFS, stores tuples (u, v)
    Q = deque()

    # Initialize for length 0 paths (i to i)
    # A path of length 0 is just a single vertex. The label sequence is empty, which is a palindrome.
    for i in range(N):
        dist[i][i] = 0
        Q.append((i, i))

    # Initialize for length 1 paths (i to j with a direct edge)
    # A path of length 1 from i to j exists if C[i][j] != '-'.
    # The label sequence is C[i][j], which is a palindrome of length 1.
    for i in range(N):
        for j in range(N):
            if C[i][j] != '-':
                # If a length 0 path exists (i==j), 0 is shorter and already set.
                # If no path or path > 1 found yet, update to 1.
                if dist[i][j] > 1:
                    dist[i][j] = 1
                    Q.append((i, j))

    # BFS
    # The state (u, v) in the queue means we have found the shortest palindrome path
    # from vertex u to vertex v, and its length is dist[u][v].
    # We can potentially find a shorter palindrome path for other pairs (p, q)
    # by extending the known palindrome path u...v.
    # If there is an edge p -> u with label c, and an edge v -> q with label c,
    # then the path p -> u -> (...palindrome u...v...) -> v -> q is a palindrome path
    # from p to q. Its length is dist[u][v] + 2.
    while Q:
        u, v = Q.popleft()
        k = dist[u][v] # Current shortest palindrome path length from u to v

        # Look for vertices p and q such that C[p][u] == C[v][q] != '-'
        # This allows extending the palindrome u...v (length k)
        # to a palindrome p...q (length k+2)
        # Iterate over all possible previous vertex p for u, and next vertex q for v.
        for p in range(N):
            # Check if there is an edge p -> u
            if C[p][u] != '-':
                char_label = C[p][u]
                # Check if there is an edge v -> q with the same label
                for q in range(N):
                    if C[v][q] != '-' and C[v][q] == char_label:
                        # Found matching edges p -> u and v -> q with label char_label
                        # This forms a palindrome path from p to q of length k + 2
                        if dist[p][q] > k + 2:
                            dist[p][q] = k + 2
                            Q.append((p, q))

    # Output the result matrix
    for i in range(N):
        row_output = []
        for j in range(N):
            if dist[i][j] == INF:
                row_output.append("-1")
            else:
                row_output.append(str(dist[i][j]))
        print(" ".join(row_output))

solve()
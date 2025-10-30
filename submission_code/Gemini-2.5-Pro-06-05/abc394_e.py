# YOUR CODE HERE
import sys
import collections

def solve():
    """
    Solves the shortest palindromic path problem on a directed graph.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read graph size and structure
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        C = [input().strip() for _ in range(N)]
    except (IOError, ValueError):
        return

    # adj[u] maps label -> list of neighbors v where u->v has that label
    adj = [collections.defaultdict(list) for _ in range(N)]
    # rev_adj[u] maps label -> list of predecessors v where v->u has that label
    rev_adj = [collections.defaultdict(list) for _ in range(N)]

    for i in range(N):
        for j in range(N):
            label = C[i][j]
            if label != '-':
                adj[i][label].append(j)
                rev_adj[j][label].append(i)

    # dist[i][j] stores the length of the shortest palindromic path from i to j
    dist = [[-1] * N for _ in range(N)]
    
    # queue for the BFS on pairs of vertices (u, v)
    queue = collections.deque()

    # Initialize with base cases: paths of length 0
    # A path i -> i with no edges has an empty label string (a palindrome).
    for i in range(N):
        if dist[i][i] == -1:
            dist[i][i] = 0
            queue.append((i, i))

    # Initialize with base cases: paths of length 1
    # A path i -> j with a single edge has a single-character label (a palindrome).
    for i in range(N):
        for j in range(N):
            if C[i][j] != '-':
                # A path of length 0 (i->i) is shorter than a self-loop of length 1.
                if dist[i][j] == -1:
                    dist[i][j] = 1
                    queue.append((i, j))

    # BFS to find longer palindromic paths
    while queue:
        u, v = queue.popleft()
        current_dist = dist[u][v]

        # Extend the palindromic path u -> ... -> v
        # to a new path: u_prev -> u -> ... -> v -> v_next.
        # This requires an edge u_prev -> u and v -> v_next with the same label.
        
        # Iterate over labels of incoming edges to u
        for label in rev_adj[u]:
            # Check for outgoing edges from v with the same label
            if label in adj[v]:
                # For each matching pair of edges, we find a new, longer palindromic path
                for u_prev in rev_adj[u][label]:
                    for v_next in adj[v][label]:
                        # If we haven't found a path to (u_prev, v_next) yet,
                        # this is the shortest one. BFS guarantees this.
                        if dist[u_prev][v_next] == -1:
                            dist[u_prev][v_next] = current_dist + 2
                            queue.append((u_prev, v_next))
    
    # Print the resulting distance matrix
    for i in range(N):
        print(*dist[i])

solve()
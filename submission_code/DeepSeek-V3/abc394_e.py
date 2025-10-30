import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    adj = []
    rev_adj = [[] for _ in range(N+1)]  # rev_adj[j][i] lists edges entering j (i -> j) with their labels
    for _ in range(N):
        line = sys.stdin.readline().strip()
        adj.append(line)
    
    # Initialize distance matrix
    INF = float('inf')
    dist = [[-1] * (N+1) for _ in range(N+1)]
    
    # We will use a queue that processes (u, v, current_palindrome_chars)
    # But tracking full strings is expensive. Instead, we can manage it as a deque or track the required characters.
    # Alternatively, for BFS, each step extends the path by 1 or 2 characters.
    # The state can be represented as (u, v, s), where s is the current required characters to form a palindrome.
    # But for BFS, we can represent s as a string that needs to be matched.
    # However, for efficiency, we can limit s to be of length 0 or 1.
    
    # The queue will store tuples of (u, v, s), and the distance.
    # Initialize the queue with (i, i, '') for all i, distance 0.
    q = deque()
    for i in range(1, N+1):
        dist[i][i] = 0
        q.append((i, i, ''))
    
    # Also, for edges i -> j, the string is c. If c is palindrome (length 1), then dist[i][j] is 1.
    for i in range(1, N+1):
        for j in range(1, N+1):
            c = adj[i-1][j-1]
            if c != '-':
                if dist[i][j] == -1 or dist[i][j] > 1:
                    dist[i][j] = 1
                    q.append((i, j, ''))
                # Also, any single character is a palindrome.
    
    # Now, process the queue.
    while q:
        u, v, s = q.popleft()
        current_dist = dist[u][v]
        if s == '':
            # We can extend by adding an edge from u and/or from v's reverse.
            # Try adding an edge from u to u2, and prepend the character.
            for u2 in range(1, N+1):
                c1 = adj[u-1][u2-1]
                if c1 == '-':
                    continue
                # New s is c1. Need to find a matching c1 from v's incoming edges.
                for v2 in range(1, N+1):
                    c2 = adj[v2-1][v-1]
                    if c2 == '-':
                        continue
                    if c1 == c2:
                        if dist[u2][v2] == -1 or dist[u2][v2] > current_dist + 2:
                            dist[u2][v2] = current_dist + 2
                            q.append((u2, v2, ''))
        else:
            # s is a single character. We need to find edges from u and v that match s.
            pass  # This part is not needed if s is always empty.
    
    # Now, handle cases where s is non-empty. But in the above BFS, s is always empty.
    # So the BFS only handles even-length palindromes. However, odd-length palindromes can be formed by paths of length 1 (single edge) or paths where the middle character is the same.
    # So, the initial BFS covers paths where the palindrome is built symmetrically. But single edges (length 1) are already handled.
    
    # Output the dist matrix.
    for i in range(1, N+1):
        row = []
        for j in range(1, N+1):
            row.append(str(dist[i][j]))
        print(' '.join(row))

solve()
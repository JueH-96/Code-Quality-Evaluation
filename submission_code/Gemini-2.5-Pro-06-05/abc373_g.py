import sys

# It is recommended to increase the recursion limit for deep recursion,
# which can occur in the DFS for bipartite matching on certain graphs.
# The default limit is often 1000, while N can be up to 300. The DFS path
# length can be at most N, so the default might be fine, but increasing it
# is safer.
sys.setrecursionlimit(2000)

def solve():
    """
    Reads input, solves the non-intersecting segments problem, and prints the result.
    """
    readline = sys.stdin.readline
    
    try:
        N = int(readline())
        # Read P points (A_i, B_i)
        P = [tuple(map(int, readline().split())) for _ in range(N)]
        # Read Q points (C_i, D_i)
        Q = [tuple(map(int, readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Handles malformed or empty input, not expected in this problem
        # but good practice.
        return

    # --- Geometric Helper Function ---
    def orientation(p, q, r):
        """
        Calculates the orientation of the ordered triplet (p, q, r).
        Returns:
         > 0 if the points make a counter-clockwise (left) turn.
         < 0 if the points make a clockwise (right) turn.
           0 if the points are collinear.
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    # --- Bipartite Graph Construction ---
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            p_i = P[i]
            q_j = Q[j]
            
            p_left_count = 0
            # Count P points to the "left" of the directed line from P_i to Q_j.
            for k in range(N):
                if i == k:
                    continue
                if orientation(p_i, q_j, P[k]) == 1:
                    p_left_count += 1
            
            q_left_count = 0
            # Count Q points to the "left" of the directed line from P_i to Q_j.
            for k in range(N):
                if j == k:
                    continue
                if orientation(p_i, q_j, Q[k]) == 1:
                    q_left_count += 1
            
            # If the counts match, the partition is balanced. Add an edge.
            if p_left_count == q_left_count:
                adj[i].append(j)

    # --- Bipartite Matching using DFS to find augmenting paths ---
    # matchR[j] stores the P-node index `i` matched with Q-node index `j`.
    matchR = [-1] * N
    
    def dfs_match(u, visited, current_adj, current_matchR):
        """
        A DFS-based function to find an augmenting path starting from P-node `u`.
        """
        for v in current_adj[u]:  # For each Q-node `v` adjacent to P-node `u`
            if not visited[v]:
                visited[v] = True
                # If Q-node `v` is unmatched, or its current partner can find a new match
                if current_matchR[v] < 0 or dfs_match(current_matchR[v], visited, current_adj, current_matchR):
                    current_matchR[v] = u
                    return True
        return False

    match_count = 0
    # Iterate through all P-nodes and try to find a match for them.
    for i in range(N):
        visited = [False] * N  # Reset visited for each augmenting path search
        if dfs_match(i, visited, adj, matchR):
            match_count += 1
            
    # --- Output ---
    if match_count < N:
        # If the size of the maximum matching is less than N, no perfect matching exists.
        print(-1)
    else:
        # A perfect matching was found. Convert it to the required permutation format.
        # R[i] = j+1 means P_i is matched with Q_j.
        # matchR[j] = i means Q_j is matched with P_i.
        R = [0] * N
        for j in range(N):
            i = matchR[j]
            # The problem asks for 1-based indexing for the permutation R.
            R[i] = j + 1
        
        print(*R)

solve()
import sys

# Increase recursion limit for deep DFS calls in SCC algorithm
# N can be up to 200,000, 2N nodes, so recursion depth can be 2N.
sys.setrecursionlimit(2 * 10**5 + 500) 

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # If A contains no 0, it is already "good". Any string S works.
    if all(a == 1 for a in A):
        print("Yes")
        return

    # 2-SAT formulation
    # A variable x_k represents S_k = 'R'.
    # Node mapping:
    # x_k  -> 2 * k
    # not x_k -> 2 * k + 1
    # Total 2N nodes.

    num_nodes = 2 * N
    
    # Adjacency lists for the implication graph and its transpose
    graph = [[] for _ in range(num_nodes)]
    graph_rev = [[] for _ in range(num_nodes)]

    def add_edge(u, v):
        """Adds a directed edge from u to v in the graph and its transpose."""
        graph[u].append(v)
        graph_rev[v].append(u)

    # Clause type 1: No two adjacent 'R's in S.
    # This means: (not x_k) OR (not x_{k+1}) for all k (cyclically).
    # Implies: x_k => (not x_{k+1}) AND x_{k+1} => (not x_k)
    for k in range(N):
        k_node = 2 * k          # Node for x_k
        not_k_node = 2 * k + 1  # Node for not x_k
        
        kp1_idx = (k + 1) % N   # Index for k+1 (cyclically)
        kp1_node = 2 * kp1_idx
        not_kp1_node = 2 * kp1_idx + 1

        # Add edges for (x_k => not x_{k+1})
        add_edge(k_node, not_kp1_node)
        # Add edges for (x_{k+1} => not x_k)
        add_edge(kp1_node, not_k_node)

    # Clause type 2: If A_j = 0, then S_j='R' OR S_{j+1}='R'.
    # This means: x_j OR x_{j+1} for relevant j (cyclically).
    # Implies: (not x_j) => x_{j+1} AND (not x_{j+1}) => x_j
    for j in range(N):
        if A[j] == 0:
            j_node = 2 * j
            not_j_node = 2 * j + 1
            
            jp1_idx = (j + 1) % N
            jp1_node = 2 * jp1_idx
            not_jp1_node = 2 * jp1_idx + 1

            # Add edges for (not x_j => x_{j+1})
            add_edge(not_j_node, jp1_node)
            # Add edges for (not x_{j+1} => x_j)
            add_edge(not_jp1_node, j_node)

    # Kosaraju's Algorithm for finding SCCs

    # Phase 1: DFS on the original graph to compute finishing times
    visited = [False] * num_nodes
    order = [] # Stores nodes in increasing order of their finishing times

    def dfs1(u):
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(num_nodes):
        if not visited[i]:
            dfs1(i)

    # Phase 2: DFS on the transpose graph
    scc_id = [-1] * num_nodes # Stores the SCC ID for each node
    current_scc = 0 # Counter for SCC IDs

    def dfs2(u):
        scc_id[u] = current_scc
        for v in graph_rev[u]:
            if scc_id[v] == -1: # If node not yet assigned to an SCC
                dfs2(v)

    order.reverse() # Process nodes in decreasing order of finishing times
    for u in order:
        if scc_id[u] == -1: # If node not yet assigned to an SCC
            dfs2(u)
            current_scc += 1

    # Check for conflicts: if x_k and not x_k are in the same SCC
    for k in range(N):
        if scc_id[2 * k] == scc_id[2 * k + 1]:
            print("No") # Conflict found, 2-SAT is unsatisfiable
            return

    # No conflicts found, 2-SAT is satisfiable
    print("Yes")

solve()
# YOUR CODE HERE
import sys

def solve():
    """
    This function solves the problem by modeling it as a graph problem.
    The main steps are:
    1.  Construct a directed graph where letters are vertices and product names are edges.
    2.  Decompose the graph into weakly connected components (WCCs).
    3.  For each WCC, find its strongly connected components (SCCs).
    4.  Construct the condensation graph (a DAG of SCCs) for the WCC.
    5.  The minimum number of paths required for the WCC is the maximum of the
        number of source SCCs and sink SCCs in its condensation graph.
    6.  The total answer is the sum of paths required for each WCC.
    """
    # It's a small graph (26 nodes), but good practice for competitive programming
    # to handle potentially deep recursion in DFS.
    sys.setrecursionlimit(2000)

    try:
        line = sys.stdin.readline()
        if not line.strip():
            return
        N = int(line)
        if N == 0:
            print(0)
            return
        used_names = {sys.stdin.readline().strip() for _ in range(N)}
    except (IOError, ValueError):
        # Handles empty input or malformed N
        return

    V = 26  # 'A' to 'Z'
    adj = [[] for _ in range(V)]
    rev_adj = [[] for _ in range(V)]
    undir_adj = [[] for _ in range(V)]
    nodes = set()

    for name in used_names:
        u = ord(name[0]) - ord('A')
        v = ord(name[1]) - ord('A')
        adj[u].append(v)
        rev_adj[v].append(u)
        nodes.add(u)
        nodes.add(v)
    
    # Build undirected graph for finding WCCs
    for u in nodes:
        for v in adj[u]:
            undir_adj[u].append(v)
            undir_adj[v].append(u)

    # --- Kosaraju's algorithm for SCCs on the whole graph ---
    order = []
    visited_kosaraju = [False] * V
    def dfs1(u):
        visited_kosaraju[u] = True
        for v_ in adj[u]:
            if not visited_kosaraju[v_]:
                dfs1(v_)
        order.append(u)

    for i in range(V):
        if i in nodes and not visited_kosaraju[i]:
            dfs1(i)

    scc_map = [-1] * V
    scc_count = 0
    def dfs2(u):
        scc_map[u] = scc_count
        for v_ in rev_adj[u]:
            if scc_map[v_] == -1:
                dfs2(v_)

    while order:
        u = order.pop()
        if scc_map[u] == -1:
            dfs2(u)
            scc_count += 1
    
    # --- Find WCCs and analyze each ---
    total_paths = 0
    wcc_visited = [False] * V
    
    for i in range(V):
        if i in nodes and not wcc_visited[i]:
            # Found a new WCC, explore it using BFS
            wcc_nodes = []
            q = [i]
            wcc_visited[i] = True
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                wcc_nodes.append(u)
                for v_ in undir_adj[u]:
                    if not wcc_visited[v_]:
                        wcc_visited[v_] = True
                        q.append(v_)
            
            # WCC found, containing nodes in wcc_nodes.
            # Identify the SCCs that are part of this WCC.
            sccs_in_wcc = set(scc_map[node] for node in wcc_nodes)
            num_sccs = len(sccs_in_wcc)
            
            if num_sccs == 1:
                total_paths += 1
                continue

            # --- Build condensation graph for the current WCC ---
            # Remap global SCC IDs to local 0..num_sccs-1 for this WCC
            scc_global_to_local = {global_id: i for i, global_id in enumerate(sccs_in_wcc)}

            scc_in_degree = [0] * num_sccs
            scc_out_degree = [0] * num_sccs
            scc_edges = set()

            for u in wcc_nodes:
                global_scc_u = scc_map[u]
                local_scc_u = scc_global_to_local[global_scc_u]
                for v in adj[u]:
                    # Check if neighbor v is in the same WCC
                    if scc_map[v] in sccs_in_wcc:
                        global_scc_v = scc_map[v]
                        if global_scc_u != global_scc_v:
                            local_scc_v = scc_global_to_local[global_scc_v]
                            if (local_scc_u, local_scc_v) not in scc_edges:
                                scc_out_degree[local_scc_u] += 1
                                scc_in_degree[local_scc_v] += 1
                                scc_edges.add((local_scc_u, local_scc_v))

            num_sources = sum(1 for j in range(num_sccs) if scc_in_degree[j] == 0)
            num_sinks = sum(1 for j in range(num_sccs) if scc_out_degree[j] == 0)
            
            total_paths += max(num_sources, num_sinks)
            
    print(total_paths)

if __name__ == "__main__":
    solve()
import sys
sys.setrecursionlimit(10000)

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    edges = []
    nodes_set = set()
    # Read edges and record nodes that appear
    for _ in range(N):
        s = input().strip()
        u = ord(s[0]) - ord('A')
        v = ord(s[1]) - ord('A')
        edges.append((u, v))
        nodes_set.add(u)
        nodes_set.add(v)
    # Build undirected adjacency for weak connectivity
    und_adj = {u: set() for u in nodes_set}
    for u, v in edges:
        und_adj[u].add(v)
        und_adj[v].add(u)
    visited = set()
    answer = 0
    # Process each weakly connected component
    for start in nodes_set:
        if start in visited:
            continue
        # BFS/DFS to collect the component's nodes
        stack = [start]
        visited.add(start)
        comp_nodes = []
        while stack:
            u = stack.pop()
            comp_nodes.append(u)
            for w in und_adj[u]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        comp_nodes_set = set(comp_nodes)
        # Collect edges in this component
        comp_edges = [(u, v) for (u, v) in edges if u in comp_nodes_set]
        # Build directed adjacency for SCC
        comp_adj = {u: [] for u in comp_nodes}
        for u, v in comp_edges:
            comp_adj[u].append(v)
        # Tarjan's algorithm to find SCCs in comp
        index = {}
        lowlink = {}
        onstack = set()
        stack_tarjan = []
        scc_id = {}
        idx = 0
        scc_count = 0

        def strongconnect(u):
            nonlocal idx, scc_count
            index[u] = idx
            lowlink[u] = idx
            idx += 1
            stack_tarjan.append(u)
            onstack.add(u)
            for w in comp_adj[u]:
                if w not in index:
                    strongconnect(w)
                    lowlink[u] = min(lowlink[u], lowlink[w])
                elif w in onstack:
                    lowlink[u] = min(lowlink[u], index[w])
            # If u is root of SCC
            if lowlink[u] == index[u]:
                # Pop stack until u
                while True:
                    w = stack_tarjan.pop()
                    onstack.remove(w)
                    scc_id[w] = scc_count
                    if w == u:
                        break
                scc_count += 1

        for u in comp_nodes:
            if u not in index:
                strongconnect(u)
        # Build condensation DAG degrees
        cond_out = [0] * scc_count
        cond_in = [0] * scc_count
        for u, v in comp_edges:
            cu = scc_id[u]
            cv = scc_id[v]
            if cu != cv:
                cond_out[cu] += 1
                cond_in[cv] += 1
        # Compute sum of out-degrees of source SCCs and max out-degree
        source_sum = 0
        for c in range(scc_count):
            if cond_in[c] == 0:
                source_sum += cond_out[c]
        max_out = 0
        for c in range(scc_count):
            if cond_out[c] > max_out:
                max_out = cond_out[c]
        # At least one string per component with edges
        k_comp = max(1, source_sum, max_out)
        answer += k_comp

    print(answer)

if __name__ == "__main__":
    main()
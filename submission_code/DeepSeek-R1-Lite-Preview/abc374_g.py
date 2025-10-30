def main():
    import sys
    sys.setrecursionlimit(1000000)
    from collections import defaultdict, deque

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    product_names = N_and_rest[1:N+1]

    # Map letters to integers
    letter_to_idx = {chr(ord('A') + i): i for i in range(26)}
    idx_to_letter = {i: chr(ord('A') + i) for i in range(26)}

    # Build graph
    graph = [[] for _ in range(26)]
    for s in product_names:
        u = letter_to_idx[s[0]]
        v = letter_to_idx[s[1]]
        graph[u].append(v)

    # Tarjan's algorithm for SCCs
    index = 0
    indices = [None] * 26
    low = [None] * 26
    on_stack = [False] * 26
    S = []
    sccs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        S.append(v)
        on_stack[v] = True
        for w in graph[v]:
            if indices[w] is None:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack[w]:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            scc = []
            while True:
                w = S.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(26):
        if indices[v] is None:
            strongconnect(v)

    # Build condensation graph
    num_scc = len(sccs)
    scc_id = [0] * 26
    for i, scc in enumerate(sccs):
        for node in scc:
            scc_id[node] = i

    cond_graph = [[] for _ in range(num_scc)]
    in_deg = [0] * num_scc
    out_deg = [0] * num_scc

    for u in range(26):
        for v in graph[u]:
            if scc_id[u] != scc_id[v]:
                cond_graph[scc_id[u]].append(scc_id[v])
                out_deg[scc_id[u]] += 1
                in_deg[scc_id[v]] += 1

    # Identify source components (no incoming edges from other components)
    source_components = [i for i in range(num_scc) if in_deg[i] == 0]

    # Calculate excess out-degree for each SCC
    excess = [out_deg[i] - in_deg[i] for i in range(num_scc)]

    # Count the number of SCCs with positive excess out-degree
    positive_excess = sum(1 for e in excess if e > 0)

    # Minimum number of strings is the number of source components plus the number of SCCs with positive excess
    min_strings = len(source_components) + positive_excess

    print(min_strings)

if __name__ == "__main__":
    main()
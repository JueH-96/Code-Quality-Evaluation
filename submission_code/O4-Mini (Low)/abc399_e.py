def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline
    
    N = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    # Step 1: Build a consistent mapping from chars in S to chars in T
    mapping = {}  # mapping[x] = y means x->y
    for s_char, t_char in zip(S, T):
        if s_char == t_char:
            continue
        if s_char in mapping:
            if mapping[s_char] != t_char:
                print(-1)
                return
        else:
            mapping[s_char] = t_char
    
    # Step 2: Build directed graph from mapping edges
    # We'll only have up to 26 nodes (a..z)
    # Count edges
    edges = len(mapping)
    
    # adjacency lists
    adj = {ch: [] for ch in mapping}
    radj = {ch: [] for ch in mapping}
    for x, y in mapping.items():
        adj[x].append(y)
        # Ensure y is in the node set, even if it has no outgoing edges
        if y not in adj:
            adj[y] = []
            radj[y] = []
        radj[y].append(x)
    
    # Step 3: Find strongly connected components (SCCs) via Kosaraju
    visited = set()
    order = []
    def dfs1(u):
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                dfs1(v)
        order.append(u)
    
    for node in adj:
        if node not in visited:
            dfs1(node)
    
    visited.clear()
    scc_id = {}
    cur_id = 0
    def dfs2(u):
        stack = [u]
        comp = []
        visited.add(u)
        while stack:
            x = stack.pop()
            comp.append(x)
            for v in radj[x]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return comp
    
    sccs = []
    for u in reversed(order):
        if u not in visited:
            comp = dfs2(u)
            sccs.append(comp)
    
    # Count how many SCCs represent real cycles (size > 1)
    cycles = 0
    for comp in sccs:
        if len(comp) > 1:
            cycles += 1
    
    # Step 4: Check for spare letter to break cycles if any
    if cycles > 0:
        used_in_S = set(S)
        # find any lowercase letter not in S
        spare_found = False
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c not in used_in_S:
                spare_found = True
                break
        if not spare_found:
            print(-1)
            return
    
    # Step 5: result = total edges + one extra operation per cycle
    print(edges + cycles)


if __name__ == "__main__":
    main()
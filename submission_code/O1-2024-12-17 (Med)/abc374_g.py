def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    used_edges = input_data[1:]  # Each is a 2-letter string, e.g. "AB"

    # We'll map 'A'..'Z' to 0..25
    def idx(ch):
        return ord(ch) - ord('A')

    # Adjacency for directed graph
    outdeg = [0]*26
    indeg = [0]*26

    # For finding undirected components, we'll build an undirected adjacency
    undirected_adj = [[] for _ in range(26)]
    used_vertex = [False]*26  # Marks if a vertex is involved in at least one used edge

    for edge in used_edges:
        u = idx(edge[0])
        v = idx(edge[1])
        outdeg[u] += 1
        indeg[v] += 1
        used_vertex[u] = True
        used_vertex[v] = True
        # Add undirected links so we can find connected components (ignoring direction)
        undirected_adj[u].append(v)
        undirected_adj[v].append(u)

    # Find connected components in the "undirected" sense, but only among vertices used_vertex == True
    visited = [False]*26

    def dfs(start):
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in undirected_adj[cur]:
                if (not visited[nxt]) and used_vertex[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        return comp

    answer = 0

    for v in range(26):
        if used_vertex[v] and not visited[v]:
            component = dfs(v)  # all vertices in this connected component

            # Count how many edges are actually in this component
            # (sum of outdeg of its vertices) tells how many directed edges are in it
            comp_outdeg = 0
            comp_indeg = 0
            for c in component:
                comp_outdeg += outdeg[c]
                comp_indeg += indeg[c]
            if comp_outdeg == 0:
                # No edges in this subgraph (perhaps a lone vertex that never appears
                # in a 2-letter code, though the problem states each S_i is used... 
                # but let's be safe). Then it needs 0 strings.
                continue

            # Count "sources" (in_degree=0 but out_degree>0) and "sinks" (out_degree=0 but in_degree>0)
            # only among vertices in this component
            sources = 0
            sinks = 0
            for c in component:
                if outdeg[c] > 0 and indeg[c] == 0:
                    sources += 1
                if indeg[c] > 0 and outdeg[c] == 0:
                    sinks += 1

            # At least one string is needed if there are edges
            # The formula for how many strings needed in this component:
            #   max(1, sources, sinks)
            answer += max(1, sources, sinks)

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()
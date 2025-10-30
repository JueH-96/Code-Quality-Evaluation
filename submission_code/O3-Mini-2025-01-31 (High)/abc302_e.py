def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    # Each vertex starts with degree 0 and no neighbors.
    deg = [0] * (n + 1)
    neighbors = [set() for _ in range(n + 1)]
    
    # Initially, every vertex is isolated.
    isolated_count = n
    output = []
    
    for _ in range(q):
        typ = next(it)
        if typ == b'1':  # Query of type "1 u v": add an edge between u and v.
            u = int(next(it))
            v = int(next(it))
            if deg[u] == 0:
                isolated_count -= 1  # u stops being isolated.
            if deg[v] == 0:
                isolated_count -= 1  # v stops being isolated.
            deg[u] += 1
            deg[v] += 1
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        else:  # Query of type "2 v": remove all edges attached to vertex v.
            v = int(next(it))
            # Iterate over a snapshot of neighbors since we will modify the set.
            for u in list(neighbors[v]):
                neighbors[u].discard(v)
                deg[u] -= 1
                if deg[u] == 0:
                    isolated_count += 1  # u becomes isolated.
            # If v had any edges, then after removal, v becomes isolated.
            if deg[v] != 0:
                isolated_count += 1
            deg[v] = 0
            neighbors[v].clear()
            
        output.append(str(isolated_count))
    
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()
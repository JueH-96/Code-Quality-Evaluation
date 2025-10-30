def main():
    import sys, collections
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    N = int(data[0])
    # There are 26 letters (vertices 0...25 corresponding to A...Z)
    outdeg = [0] * 26
    indeg = [0] * 26
    # For connectivity (undirected) we keep a set for each vertex
    undirected = [set() for _ in range(26)]
    used = [False] * 26
  
    # Process each used product name (each two-letter uppercase string)
    index = 1
    for i in range(N):
        s = data[index]
        index += 1
        u = ord(s[0]) - ord('A')
        v = ord(s[1]) - ord('A')
        used[u] = True
        used[v] = True
        outdeg[u] += 1
        indeg[v] += 1
        undirected[u].add(v)
        undirected[v].add(u)
  
    visited = [False] * 26
    total_strings = 0
  
    # For every vertex that appears in at least one used name
    for i in range(26):
        if used[i] and not visited[i]:
            # Do a BFS (or DFS) in the undirected graph to get one connected component.
            comp = []
            q = collections.deque()
            q.append(i)
            visited[i] = True
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in undirected[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            # In this component count how many vertices have zero incoming and zero outgoing.
            zero_in = 0
            zero_out = 0
            for v in comp:
                if indeg[v] == 0:
                    zero_in += 1
                if outdeg[v] == 0:
                    zero_out += 1
            comp_needed = max(zero_in, zero_out)
            if comp_needed == 0:
                comp_needed = 1  # If every vertex has at least one incoming and one outgoing, one NG-string suffices.
            total_strings += comp_needed
    sys.stdout.write(str(total_strings))
  
if __name__ == '__main__':
    main()
def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(300000)
    
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    idx = 1
    # P[i] will hold the list of books you must read before reading book i.
    P = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = int(data[idx])
        idx += 1
        if c:
            P[i] = list(map(int, data[idx: idx + c]))
            idx += c
        else:
            P[i] = []
    
    # -----------------------------------------------------------------------------
    # Step 1. Determine the minimal set of books required (transitively) to read book 1.
    #
    # Since book 1 requires you to have read all books in P[1], and each of those books
    # in turn has its own prerequisites, we use a DFS (implemented iteratively) to
    # get the transitive closure of prerequisites for book 1.
    #
    # The minimal set (excluding book 1) is uniquely determined.
    # -----------------------------------------------------------------------------
    
    needed = set()
    stack = []
    # Start with the direct prerequisites of book1.
    for d in P[1]:
        if d not in needed:
            needed.add(d)
            stack.append(d)
    
    while stack:
        cur = stack.pop()
        for d in P[cur]:
            if d not in needed:
                needed.add(d)
                stack.append(d)
                
    # -----------------------------------------------------------------------------
    # Step 2. Produce a valid reading order (topological ordering) for the books 
    # in the minimal set.
    # 
    # In our dependency graph, an edge goes from each prerequisite to the book that depends 
    # on it. We then perform a Kahn’s algorithm–based topological sort on the induced subgraph.
    # -----------------------------------------------------------------------------
    
    # Build dependency graph and indegree for nodes in 'needed'.
    graph = {b: [] for b in needed}
    indegree = {b: 0 for b in needed}
    for b in needed:
        for prereq in P[b]:
            if prereq in needed:  # only consider dependencies within our minimal set
                graph[prereq].append(b)
                indegree[b] += 1

    dq = deque()
    # all nodes with no incoming edges (i.e. no prerequisites inside the set)
    for b in needed:
        if indegree[b] == 0:
            dq.append(b)
    order = []
    while dq:
        cur = dq.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                dq.append(nxt)
                
    # order now is one valid order in which to read just the required books.
    # Book 1 may then be read after all of these.
    sys.stdout.write(" ".join(map(str, order)) + "
")
    
if __name__ == '__main__':
    main()
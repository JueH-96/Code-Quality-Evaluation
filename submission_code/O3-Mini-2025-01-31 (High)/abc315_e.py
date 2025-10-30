def main():
    import sys
    from collections import deque

    # Read all input tokens.
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)

    N = int(next(it))
    # For each book i (1-indexed), store its prerequisites.
    prereq = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        c = int(next(it))
        if c:
            # The next c tokens are the prerequisites for book i.
            prereq[i] = [int(next(it)) for _ in range(c)]
        else:
            prereq[i] = []
    
    # We want the minimum set of books you must read to be able to read book 1.
    # That set is exactly the transitive closure of prerequisites starting at book 1.
    visited = [False] * (N+1)
    stack = [1]
    while stack:
        cur = stack.pop()
        if visited[cur]:
            continue
        visited[cur] = True
        for p in prereq[cur]:
            if not visited[p]:
                stack.append(p)
    # The required set (including book 1) consists of all books visited.
    req_set = {i for i in range(1, N+1) if visited[i]}

    # Build a dependency graph for only the required books.
    # Recall: if book i requires book p, we draw an edge p -> i.
    graph = [[] for _ in range(N+1)]
    in_degree = [0]*(N+1)
    for i in req_set:
        for p in prereq[i]:
            # By the closure property, every prerequisite p of a required book i is also required.
            if p in req_set:
                graph[p].append(i)
                in_degree[i] += 1

    # Compute a topological order of the required subgraph using Kahn's algorithm.
    dq = deque()
    for i in req_set:
        if in_degree[i] == 0:
            dq.append(i)
            
    topo = []
    while dq:
        u = dq.popleft()
        topo.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                dq.append(v)
                
    # The valid reading order is given by the topological order.
    # We must output the reading order of exactly those books (other than book 1)
    # that are required to read book 1.
    result = [str(book) for book in topo if book != 1]
    sys.stdout.write(" ".join(result))

    
if __name__ == '__main__':
    main()
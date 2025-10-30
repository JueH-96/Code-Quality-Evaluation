def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # We'll parse the rest of the input_data from index=1 onward
    idx = 1

    # Build adjacency (dep -> book) and reverse adjacency (book -> dep)
    # where an edge p->i means: to read book i, you must have read p first.
    adj = [[] for _ in range(N+1)]
    rev = [[] for _ in range(N+1)]

    for i in range(1, N+1):
        Ci = int(input_data[idx])
        idx += 1
        for _ in range(Ci):
            p = int(input_data[idx])
            idx += 1
            # p -> i in original graph
            adj[p].append(i)
            # i -> p in reverse graph
            rev[i].append(p)

    # 1) Find all books needed to read book 1 by BFS in the reversed graph
    visited = [False]*(N+1)
    visited[1] = True
    queue = deque([1])

    while queue:
        u = queue.popleft()
        for v in rev[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    # 2) Topological sort of the subgraph consisting of visited nodes
    in_degree = [0]*(N+1)
    for u in range(1, N+1):
        if visited[u]:
            for w in adj[u]:
                if visited[w]:
                    in_degree[w] += 1

    zero_dq = deque()
    for u in range(1, N+1):
        if visited[u] and in_degree[u] == 0:
            zero_dq.append(u)

    topo_order = []
    while zero_dq:
        u = zero_dq.popleft()
        topo_order.append(u)
        for w in adj[u]:
            if visited[w]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    zero_dq.append(w)

    # 3) Remove book 1 from the final order and print the rest
    #    (book 1 is always in visited, but we don't print it)
    answer = [str(x) for x in topo_order if x != 1]
    print(" ".join(answer))

# Do not forget to call main()
main()
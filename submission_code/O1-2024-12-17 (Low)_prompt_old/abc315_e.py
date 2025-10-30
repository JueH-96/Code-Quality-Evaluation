def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    # adjacency: book u -> list of books v that depend on u (original direction)
    G = [[] for _ in range(N+1)]
    # reverse adjacency: book v -> list of books u for which u is a prerequisite (v -> u in the reverse graph)
    R = [[] for _ in range(N+1)]

    idx = 1
    for i in range(1, N+1):
        Ci = int(input_data[idx])
        idx += 1
        for _ in range(Ci):
            p = int(input_data[idx])
            idx += 1
            # p -> i in the original graph
            G[p].append(i)
            # i -> p in the reverse graph
            R[i].append(p)

    # 1) Find all ancestors of book 1 (including 1) in the original graph
    #    by traversing from 1 in the reverse graph.
    from collections import deque
    visited = [False]*(N+1)
    visited[1] = True
    queue = deque([1])

    while queue:
        cur = queue.popleft()
        for pre in R[cur]:
            if not visited[pre]:
                visited[pre] = True
                queue.append(pre)

    # 2) We only need to topologically sort the subgraph of visited nodes
    #    in the original direction.
    #    Compute in-degrees restricted to those visited nodes.
    in_degree = [0]*(N+1)
    for u in range(1, N+1):
        if visited[u]:
            for v in G[u]:
                if visited[v]:
                    in_degree[v] += 1

    # 3) Topological sort among visited nodes
    topo_order = []
    q = deque()
    for i in range(1, N+1):
        if visited[i] and in_degree[i] == 0:
            q.append(i)

    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in G[u]:
            if visited[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

    # 4) Exclude book 1, then print the rest in the topological order
    #    The problem states if there are multiple orders, any is fine.
    answer = [str(x) for x in topo_order if x != 1]
    print(" ".join(answer))

def main():
    solve()

if __name__ == "__main__":
    main()
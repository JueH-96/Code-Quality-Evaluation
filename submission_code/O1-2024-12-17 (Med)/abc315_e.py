def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    idx = 1

    # rev_adj[i] will hold the list of direct prerequisites of book i
    rev_adj = [[] for _ in range(N+1)]
    
    # Read input
    for i in range(1, N+1):
        C_i = int(input_data[idx]); idx += 1
        prerequisites = input_data[idx:idx + C_i]
        idx += C_i
        # Store prerequisites in reverse adjacency:
        # i -> each p in prerequisites (so from i we can go to p)
        # because in the forward direction p -> i
        rev_adj[i] = list(map(int, prerequisites))


    # Step 1: Find all ancestors of book 1 (including 1) by BFS/DFS on rev_adj
    visited = [False]*(N+1)
    visited[1] = True
    queue = deque([1])
    while queue:
        curr = queue.popleft()
        for p in rev_adj[curr]:
            if not visited[p]:
                visited[p] = True
                queue.append(p)

    # Step 2: Build forward adjacency for the subgraph
    # adjacency[p] = list of i (books) for which p is a prerequisite
    adjacency = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        for p in rev_adj[i]:
            adjacency[p].append(i)

    # Step 3: Compute in-degrees (restricted to visited set)
    in_degree = [0]*(N+1)
    for p in range(1, N+1):
        if visited[p]:
            for i in adjacency[p]:
                if visited[i]:
                    in_degree[i] += 1

    # Step 4: Topological sort on visited subgraph
    topo_order = []
    dq = deque()
    for book in range(1, N+1):
        if visited[book] and in_degree[book] == 0:
            dq.append(book)

    while dq:
        u = dq.popleft()
        topo_order.append(u)
        for v in adjacency[u]:
            if visited[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    dq.append(v)

    # Remove book 1 from the final order (we only print prerequisites)
    # If book 1 ended up in the order, remove it; it is guaranteed visited[1] = True
    # so book 1 should appear in topo_order.
    # The problem states we should exclude book 1 from the output.
    if 1 in topo_order:
        topo_order.remove(1)

    # Output result
    print(" ".join(map(str, topo_order)))


# Do not forget to call main() 
if __name__ == "__main__":
    main()
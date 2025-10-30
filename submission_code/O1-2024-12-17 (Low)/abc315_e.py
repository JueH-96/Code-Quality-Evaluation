def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    # We'll build two adjacency lists:
    # 1) forward[i] will list all books that depend on book i (i.e., i -> forward[i][...])
    # 2) reverse[i] will list all books which are prerequisites of book i (i.e., reverse[i] are the direct prereqs of i)
    # Then, 1) is used for topological sort, and 2) is used for finding ancestors of book 1.
    forward = [[] for _ in range(N+1)]
    reverse_ = [[] for _ in range(N+1)]

    idx = 1
    for i in range(1, N+1):
        C_i = int(input_data[idx]); idx += 1
        for _ in range(C_i):
            p = int(input_data[idx]); idx += 1
            # p -> i in the forward sense
            forward[p].append(i)
            # i -> p in the reverse sense (i depends on p)
            reverse_[i].append(p)

    # Step 1: Find all ancestors of book 1 (including book 1 itself),
    # by traversing from 1 in the reversed graph.
    from collections import deque
    queue = deque([1])
    visited = set([1])
    while queue:
        cur = queue.popleft()
        for pre in reverse_[cur]:
            if pre not in visited:
                visited.add(pre)
                queue.append(pre)

    # Step 2: Topological sort the subgraph induced by visited in the forward direction.
    # We'll count in-degrees within this set only.
    in_degree = [0]*(N+1)
    for v in visited:
        for nxt in forward[v]:
            if nxt in visited:
                in_degree[nxt] += 1

    # Kahn's Algorithm
    topo_order = []
    queue = deque()
    # Enqueue all nodes in visited with in_degree 0
    for v in visited:
        if in_degree[v] == 0:
            queue.append(v)

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Decrement in_degree of all forward neighbors in visited
        for nxt in forward[node]:
            if nxt in visited:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

    # Now topo_order is a valid topological ordering of all ancestors, including 1.
    # We remove 1 from the printing, since the problem wants to exclude book 1.
    # We only want to print the books that are strictly needed (visited) and != 1, in topological order.
    # The set "visited" is the minimal set of ancestors plus book 1.
    # Sort order is already topological, so just filter out "1".
    answer = [x for x in topo_order if x != 1]
    print(" ".join(map(str, answer)))

# Do not forget to call main()!
if __name__ == "__main__":
    main()
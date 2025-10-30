def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    a = list(map(int, input_data[1:]))

    # 1) Identify the cycle size for each node (0 if not on a cycle)
    # We'll use an iterative "DFS-like" approach with a stack to find cycles.
    visited = [0]*(N+1)  # 0 = unvisited, 1 = visiting, 2 = done
    stack_index = [-1]*(N+1)
    cycle_size = [0]*(N+1)  # For nodes in a cycle, store the cycle length; else 0.

    for start in range(1, N+1):
        if visited[start] == 0:
            curr = start
            local_stack = []
            while True:
                if visited[curr] == 0:
                    visited[curr] = 1
                    stack_index[curr] = len(local_stack)
                    local_stack.append(curr)
                    curr = a[curr-1]  # next node (adjust for 1-based index)
                elif visited[curr] == 1:
                    # Found a cycle
                    start_idx = stack_index[curr]
                    c_len = len(local_stack) - start_idx
                    # Mark cycle members
                    for i in range(start_idx, len(local_stack)):
                        cycle_node = local_stack[i]
                        cycle_size[cycle_node] = c_len
                    # Pop everything from the stack, mark visited=2
                    while local_stack:
                        node = local_stack.pop()
                        visited[node] = 2
                        stack_index[node] = -1
                    break
                else:
                    # visited[curr] == 2 => no new cycle
                    # Just pop all from local_stack
                    while local_stack:
                        node = local_stack.pop()
                        visited[node] = 2
                        stack_index[node] = -1
                    break

    # 2) Build reverse edges to compute the number of reachable nodes (f[u]) from the cycle outward
    rev = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        rev[a[u-1]].append(u)

    # 3) We'll store f[u] = number of reachable nodes from u
    f = [-1]*(N+1)
    from collections import deque
    q = deque()

    # Initialize f[u] for cycle nodes; push them into queue
    for u in range(1, N+1):
        if cycle_size[u] > 0:
            f[u] = cycle_size[u]
            q.append(u)

    # 4) BFS (topological order in reverse) to compute f[v] = 1 + f[u] for v->u edges
    while q:
        u = q.popleft()
        for v in rev[u]:
            if f[v] == -1:   # not computed yet
                f[v] = f[u] + 1
                q.append(v)

    # 5) Sum up all f[u] for the answer
    ans = sum(f[1:])  # f[0] is unused
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()
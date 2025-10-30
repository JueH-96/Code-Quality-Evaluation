def main():
    import sys
    sys.setrecursionlimit(3000000)
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:]))
    # Convert 1-indexed vertices to 0-indexed.
    # Each vertex i (0-indexed) has an edge to (a[i]-1).

    # Our plan:
    #   For every vertex u, the vertices reachable from u are exactly those
    #   that appear when we repeatedly follow the unique edge from u.
    #   In such a "functional graph", each component is a cycle with trees feeding into it.
    #   If u is on a tree with distance d to a cycle (which has cycle size c),
    #   then the distinct vertices reachable are d (chain nodes outside the cycle)
    #   plus the c vertices in the cycle. When u is on the cycle,
    #   its reachable nodes are the entire cycle (size c).
    #
    # We then define dp[u] as the number of reachable vertices (distinct)
    # from vertex u. So if u is not in a cycle,
    #   dp[u] = dp[a[u]-1] + 1,
    # and if u is in a cycle (or self-loop),
    #   dp[u] = cycle_size for the cycle containing u.
    #
    # To compute these values we perform a depth-first search (DFS)
    # while carefully detecting cycles. We use a three-state marking:
    #   0: unvisited, 1: visiting (in current DFS stack), 2: finished.
    #
    # Because N can be up to 200000, we use an iterative DFS.

    state = [0] * n  # 0: unvisited; 1: in stack; 2: finished
    dp = [0] * n     # dp[i] = number of reachable vertices from node i

    for i in range(n):
        if state[i] != 0:
            continue
        stack = []
        cur = i
        while True:
            if state[cur] == 0:
                state[cur] = 1
                stack.append(cur)
                nxt = a[cur] - 1
                cur = nxt
                continue
            elif state[cur] == 1:
                # Found a cycle.
                # Locate cur's first appearance in the current DFS stack.
                cycle_index = stack.index(cur)
                cycle_nodes = stack[cycle_index:]
                cycle_size = len(cycle_nodes)
                for node in cycle_nodes:
                    dp[node] = cycle_size
                    state[node] = 2  # mark these as finished.
                # Now unwind the remaining stack.
                while stack:
                    node = stack.pop()
                    if state[node] != 2:
                        dp[node] = dp[a[node]-1] + 1
                        state[node] = 2
                break
            else:  # state[cur] == 2, already computed dp for cur.
                while stack:
                    node = stack.pop()
                    dp[node] = dp[a[node]-1] + 1
                    state[node] = 2
                break

    # Compute the total number of pairs (u, v) such that v is reachable from u.
    result = sum(dp)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    # Convert to 0-based for convenience
    a = list(map(int, input().split()))
    a = [x - 1 for x in a]

    visited = [0] * N       # 0 = not visited, 1 = in stack, 2 = fully processed
    in_cycle = [False] * N  # Marks whether a node is part of a cycle
    cycle_id = [-1] * N     # Which cycle (component) each node belongs to
    stack = []
    pos_in_stack = [-1] * N

    # We'll store the size of each cycle in order
    found_cycle_sizes = []
    current_cycle_id = 0

    # First DFS to identify cycles and mark cycle nodes
    def dfs(u):
        nonlocal current_cycle_id
        visited[u] = 1
        pos_in_stack[u] = len(stack)
        stack.append(u)
        nxt = a[u]

        if visited[nxt] == 0:
            dfs(nxt)
        elif visited[nxt] == 1:
            # Found a cycle
            start = pos_in_stack[nxt]
            cycle_nodes = stack[start:]
            cycle_len = len(cycle_nodes)
            found_cycle_sizes.append(cycle_len)
            for node in cycle_nodes:
                in_cycle[node] = True
                cycle_id[node] = current_cycle_id
            # Remove the cycle part from current recursion stack
            del stack[start:]
            current_cycle_id += 1

        # Mark u fully processed
        visited[u] = 2
        stack.pop()
        pos_in_stack[u] = -1

    # Run DFS on all nodes to discover all cycles
    for i in range(N):
        if visited[i] == 0:
            dfs(i)

    # Distances to the cycle for each node
    dist = [-1] * N

    # Compute distance (number of steps) from a node to its cycle node
    def get_dist(u):
        if dist[u] != -1:
            return dist[u]
        if in_cycle[u]:
            dist[u] = 0
        else:
            dist[u] = 1 + get_dist(a[u])
        return dist[u]

    # For nodes not in a cycle, find which cycle they belong to
    def get_cycle_id(u):
        if cycle_id[u] != -1:
            return cycle_id[u]
        cycle_id[u] = get_cycle_id(a[u])
        return cycle_id[u]

    # Prepare arrays to accumulate counts
    cycle_count = len(found_cycle_sizes)  # Number of distinct cycles
    c_comp_size = [0] * cycle_count       # Component size (how many nodes map to this cycle)
    sum_dist = [0] * cycle_count          # Sum of distances for each cycle

    # Fill in dist[] and cycle_id[] for all nodes
    for i in range(N):
        d = get_dist(i)
        cid = get_cycle_id(i)
        c_comp_size[cid] += 1
        sum_dist[cid] += d

    # Now compute the total number of reachable pairs
    # formula for each cycle component:
    #   sum_dist[cid] + (cycle_size[cid] * component_size)
    answer = 0
    for cid in range(cycle_count):
        c_size = found_cycle_sizes[cid]   # size of this cycle
        comp_size = c_comp_size[cid]
        answer += sum_dist[cid] + c_size * comp_size

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()
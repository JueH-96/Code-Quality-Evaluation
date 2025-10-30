def solve():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = list(map(int, input_data[1:]))

    # Graph properties:
    # - N vertices, numbered 1..N
    # - For each i in [1..N], there is an edge i -> a_i (outdegree=1)
    # We need to count the number of pairs (u, v) such that v is reachable from u.
    #
    # Key observation:
    # Each connected component has exactly one cycle since every node has outdegree=1.
    # If we follow out edges from any vertex, it eventually hits a unique cycle.
    # From a vertex u:
    #   - we traverse a chain of zero or more edges until hitting a cycle node
    #   - from there, all nodes in that cycle are reachable
    #
    # Let cycle_size[c] = length of that cycle.
    # For a node u, define dist[u] = number of edges in the path from u to the cycle
    #   (not counting the cycle node itself).
    # Then the set of reachable nodes from u has size = dist[u] + cycle_size[u's cycle].
    #
    # So the answer = sum_{u=1..N}( dist[u] + cycle_size_of_u ).
    #
    # Steps to solve:
    # 1) Find all cycles and mark the nodes on those cycles.
    # 2) For each cycle, set dist = 0 for cycle nodes, cycle_size[node] = length of that cycle.
    # 3) Traverse in reverse (using in-edges) from each cycle node, assigning dist[u] = dist[v] + 1
    #    for each edge v->u where v's dist is known, and v leads to the same cycle.
    # 4) Sum dist[u] + cycle_size[u] for all u.

    # Build reverse edges for BFS/DFS from cycle nodes
    rev = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        rev[edges[i-1]].append(i)

    visited = [False]*(N+1)   # marks if we've fully processed the node
    in_stack = [False]*(N+1)  # marks if the node is currently in the recursion stack
    dist = [0]*(N+1)          # distance to the cycle from each node
    cycle_size = [0]*(N+1)    # size of the cycle to which the node belongs
    # We'll store all nodes that appear on cycles, which we discover

    def dfs_find_cycle(start):
        # We'll follow the chain from 'start' until we either
        # - revisit a node in_stack => found a cycle
        # - or reach a visited node => no new cycle from there
        path = []
        current = start
        while True:
            path.append(current)
            in_stack[current] = True
            visited[current] = True
            nxt = edges[current-1]
            if not visited[nxt]:
                current = nxt
            else:
                if in_stack[nxt]:
                    # found a cycle
                    # cycle is from the first occurrence of nxt in path to end
                    idx_in_path = path.index(nxt)
                    cyc_nodes = path[idx_in_path:]  # the cycle
                    cyc_len = len(cyc_nodes)
                    for cn in cyc_nodes:
                        cycle_size[cn] = cyc_len
                    # mark them as no longer in stack
                    for cn in cyc_nodes:
                        in_stack[cn] = False
                    # nodes before the cycle in path are not in a cycle
                    for cn in path[:idx_in_path]:
                        in_stack[cn] = False
                    break
                # else we either hit a node not in stack => no new cycle
                # mark all in path as not in stack
                for cn in path:
                    in_stack[cn] = False
                break

    # Find all cycles
    for i in range(1, N+1):
        if not visited[i]:
            dfs_find_cycle(i)

    # Now we do a BFS/DFS in reverse from each cycle node to assign dist and cycle_size
    from collections import deque
    q = deque()
    # initialize queue with cycle nodes
    for i in range(1, N+1):
        if cycle_size[i] > 0:  # i is on a cycle
            dist[i] = 0
            q.append(i)

    # BFS to compute dist[u] = dist[v] + 1 for u->v
    # cycle_size[u] = cycle_size[v]
    while q:
        v = q.popleft()
        for u in rev[v]:
            if cycle_size[u] == 0:  
                # means u not assigned yet 
                cycle_size[u] = cycle_size[v]
                dist[u] = dist[v] + 1
                q.append(u)

    # compute the answer
    ans = 0
    for i in range(1, N+1):
        ans += dist[i] + cycle_size[i]

    print(ans)

# Let's call solve() here
if __name__ == "__main__":
    solve()
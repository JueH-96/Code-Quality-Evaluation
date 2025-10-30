import sys


def main() -> None:
    # Read all integers at once (fast and robust against line breaks)
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    n = next(it)                          # number of vertices
    adj = [[] for _ in range(n)]          # adjacency list

    # read edges
    for _ in range(n - 1):
        a = next(it) - 1                  # convert to 0-based
        b = next(it) - 1
        adj[a].append(b)
        adj[b].append(a)

    # read weights C
    C = [next(it) for _ in range(n)]

    total_w = sum(C)                      # Σ C_i

    # ---------- first DFS (iterative) ----------
    parent = [-1] * n
    depth  = [0] * n
    order  = []                           # will store traversal order
    stack  = [0]                          # root the tree at vertex 0

    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            depth[v] = depth[u] + 1
            stack.append(v)

    subtree_w = [0] * n                   # Σ C in subtree of each node
    f_root = 0                            # f( root = 0 )

    # process vertices in reverse order to accumulate subtree weights
    for u in reversed(order):
        sw = C[u]
        for v in adj[u]:
            if v == parent[u]:
                continue
            sw += subtree_w[v]
        subtree_w[u] = sw
        f_root += C[u] * depth[u]

    # ---------- second DFS (rerooting) ----------
    best = f_root
    stack = [(0, -1, f_root)]             # (node, parent, f(value))

    while stack:
        u, p, cur_f = stack.pop()
        for v in adj[u]:
            if v == p:
                continue
            # when moving root from u to its child v
            # nodes inside v's subtree get  -1 distance, others +1
            new_f = cur_f + (total_w - 2 * subtree_w[v])
            if new_f < best:
                best = new_f
            stack.append((v, u, new_f))

    print(best)


if __name__ == "__main__":
    main()
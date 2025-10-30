import sys
from collections import deque

# ----------  auxiliary (LCA, distance)  ----------
def build_lca_and_times(n, adj, root=1):
    """
    Pre–order (DFS order) without recursion + binary–lifting tables
    returns:  tin  (entry time in DFS order)
              depth
              up    (binary–lifting table)
    """
    LOG = n.bit_length()        # enough so that 2**LOG > n
    up = [[0]*(n+1) for _ in range(LOG)]
    depth = [0]*(n+1)
    tin   = [0]*(n+1)

    timer = 0
    stack = [(root, 0)]         # (node, parent)

    while stack:
        v, p = stack.pop()

        # preorder visit
        timer += 1
        tin[v] = timer
        up[0][v] = p
        depth[v] = depth[p] + 1 if p else 0

        # fill lifting table line-by-line
        for i in range(1, LOG):
            anc = up[i-1][v]
            up[i][v] = up[i-1][anc] if anc else 0

        # push children (reversed gives same order as recursive DFS)
        for nxt in reversed(adj[v]):
            if nxt != p:
                stack.append((nxt, v))

    return tin, depth, up, LOG


def lca(u, v, depth, up, LOG):
    if depth[u] < depth[v]:
        u, v = v, u
    # lift u up so that depth[u] == depth[v]
    diff = depth[u] - depth[v]
    bit = 0
    while diff:
        if diff & 1:
            u = up[bit][u]
        diff >>= 1
        bit += 1
    if u == v:
        return u
    for k in range(LOG-1, -1, -1):
        if up[k][u] != up[k][v]:
            u = up[k][u]
            v = up[k][v]
    return up[0][u]


def distance(u, v, depth, up, LOG):
    w = lca(u, v, depth, up, LOG)
    return depth[u] + depth[v] - 2 * depth[w]


# ----------  main solution  ----------
def minimal_subtree_size(n, k, edges, terminals):
    # adjacency list
    adj = [[] for _ in range(n+1)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # prepare LCA structures
    tin, depth, up, LOG = build_lca_and_times(n, adj, root=1)

    # sort terminals by DFS entry time
    terminals.sort(key=lambda x: tin[x])

    # sum distances of consecutive vertices in the cyclic order
    total_dist = 0
    for i in range(k):
        u = terminals[i]
        v = terminals[(i+1) % k]   # wrap around
        total_dist += distance(u, v, depth, up, LOG)

    edges_in_subtree = total_dist // 2
    return edges_in_subtree + 1


# ----------  driver  ----------
def main() -> None:
    input_data = sys.stdin.readline

    N, K = map(int, input_data().split())
    edge_list = [tuple(map(int, input_data().split())) for _ in range(N-1)]
    V = list(map(int, input_data().split()))

    answer = minimal_subtree_size(N, K, edge_list, V)
    print(answer)


if __name__ == "__main__":
    main()
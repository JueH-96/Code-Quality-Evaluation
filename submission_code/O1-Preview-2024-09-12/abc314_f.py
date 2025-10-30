# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    mod = 998244353

    N = int(sys.stdin.readline())
    p = []
    q = []
    for _ in range(N - 1):
        pi, qi = map(int, sys.stdin.readline().split())
        p.append(pi - 1)  # adjust to 0-based index
        q.append(qi - 1)

    # Precompute modular inverses
    max_size = 2 * N
    inv = [0] * (max_size + 1)
    inv[1] = 1
    for i in range(2, max_size + 1):
        inv[i] = mod - mod // i * inv[mod % i] % mod

    # Build the tree
    class Node:
        def __init__(self, index=None):
            self.parent = None
            self.left = None
            self.right = None
            self.size = 1
            self.index = index  # For leaf nodes, the player index (0-based)

    nodes = [Node(i) for i in range(N)]  # Initial leaf nodes for players

    parent_nodes = []
    parent_index = N  # Indexing internal nodes beyond player indices

    parent_map = {}  # Map from node to its parent

    # Union-Find structure to keep track of team roots
    parent_uf = [i for i in range(N + N - 1)]  # Union-Find parent array
    size_uf = [1] * (N + N -1)  # Sizes of the sets

    def find(u):
        while parent_uf[u] != u:
            parent_uf[u] = parent_uf[parent_uf[u]]
            u = parent_uf[u]
        return u

    for i in range(N - 1):
        pi = p[i]
        qi = q[i]
        u = find(pi)
        v = find(qi)
        new_node = Node()
        new_node.left = nodes[u]
        new_node.right = nodes[v]
        new_node.size = size_uf[u] + size_uf[v]
        nodes.append(new_node)
        parent_uf[u] = len(nodes) - 1
        parent_uf[v] = len(nodes) - 1
        size_uf[len(nodes) - 1] = new_node.size
        nodes[u].parent = new_node
        nodes[v].parent = new_node

    # Now, for each player, traverse from leaf to root and sum per-node contributions
    E = [0] * N  # Expected wins per player
    for i in range(N):
        curr_node = nodes[i]
        while curr_node.parent is not None:
            node_size = curr_node.parent.size
            E[i] = (E[i] + inv[node_size]) % mod
            curr_node = curr_node.parent

    print(' '.join(map(str, E)))
threading.Thread(target=main).start()
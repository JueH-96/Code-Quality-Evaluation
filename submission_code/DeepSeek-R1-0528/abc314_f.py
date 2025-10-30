MOD = 998244353

import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    edges = []
    index = 1
    for i in range(n-1):
        p = int(data[index]); q = int(data[index+1]); index += 2
        edges.append((p, q))
    
    inv = [0] * (n+1)
    inv[1] = 1
    for i in range(2, n+1):
        inv[i] = (MOD - (MOD // i)) * inv[MOD % i] % MOD

    dsu = list(range(n+1))
    comp = [0] * (n+1)
    for i in range(1, n+1):
        comp[i] = i
    size_dsu = [0] * (n+1)
    for i in range(1, n+1):
        size_dsu[i] = 1

    max_nodes = 2 * n + 10
    left_child = [0] * (max_nodes)
    right_child = [0] * (max_nodes)
    tree_parent = [0] * (max_nodes)
    tree_size = [0] * (max_nodes)
    for i in range(1, n+1):
        tree_size[i] = 1

    def find(x):
        stack = []
        while dsu[x] != x:
            stack.append(x)
            x = dsu[x]
        rep = x
        for y in stack:
            dsu[y] = rep
        return rep

    next_node = n+1
    for p, q in edges:
        r1 = find(p)
        r2 = find(q)
        u = next_node
        next_node += 1

        a_node = comp[r1]
        b_node = comp[r2]
        left_child[u] = a_node
        right_child[u] = b_node
        tree_parent[a_node] = u
        tree_parent[b_node] = u
        tree_size[u] = tree_size[a_node] + tree_size[b_node]

        if size_dsu[r1] < size_dsu[r2]:
            r1, r2 = r2, r1
        dsu[r2] = r1
        size_dsu[r1] = size_dsu[r1] + size_dsu[r2]
        comp[r1] = u

    root = next_node - 1
    F = [0] * (max_nodes)
    q = deque()
    q.append(root)
    F[root] = 0

    while q:
        u = q.popleft()
        l = left_child[u]
        r = right_child[u]
        if l == 0 and r == 0:
            continue
        total_size = tree_size[u]
        if l != 0:
            term_l = tree_size[l] * inv[total_size] % MOD
            F[l] = (F[u] + term_l) % MOD
            q.append(l)
        if r != 0:
            term_r = tree_size[r] * inv[total_size] % MOD
            F[r] = (F[u] + term_r) % MOD
            q.append(r)

    ans = []
    for i in range(1, n+1):
        ans.append(str(F[i]))
    print(" ".join(ans))

if __name__ == '__main__':
    main()
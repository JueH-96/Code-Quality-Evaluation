import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    K = int(data[idx]); idx +=1

    edges = []
    for _ in range(M):
        u = int(data[idx])-1; idx +=1
        v = int(data[idx])-1; idx +=1
        w = int(data[idx]); idx +=1
        edges.append( (w, u, v) )
    A = list(map(lambda x: int(x)-1, data[idx:idx+K]))
    idx += K
    B = list(map(lambda x: int(x)-1, data[idx:idx+K]))
    idx += K

    parent = list(range(N))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u = find(u)
        v = find(v)
        if u == v:
            return False
        parent[v] = u
        return True

    edges_sorted = sorted(edges)
    mst = []
    for w, u, v in edges_sorted:
        if union(u, v):
            mst.append( (u, v, w) )
            if len(mst) == N-1:
                break

    from collections import defaultdict
    tree = defaultdict(list)
    for u, v, w in mst:
        tree[u].append( (v, w) )
        tree[v].append( (u, w) )

    LOG = 20
    up = [[(-1, -1)] * N for _ in range(LOG)]
    max_edge = [[0] * N for _ in range(LOG)]
    depth = [0] * N

    stack = [(0, -1, 0)]
    while stack:
        u, parent_u, current_max = stack.pop()
        up[0][u] = (parent_u, current_max)
        for v, w in tree[u]:
            if v != parent_u:
                depth[v] = depth[u] + 1
                new_max = max(current_max, w)
                stack.append( (v, u, new_max) )

    for k in range(1, LOG):
        for v in range(N):
            p, m = up[k-1][v]
            if p == -1:
                up[k][v] = (-1, 0)
                max_edge[k][v] = m
            else:
                up_k_minus_1, m_minus_1 = up[k-1][p]
                up[k][v] = (up_k_minus_1, m_minus_1)
                max_edge[k][v] = max(m_minus_1, m)

    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        max_e = 0
        for k in reversed(range(LOG)):
            if depth[u] - (1 << k) >= depth[v]:
                max_e = max(max_e, max_edge[k][u])
                u = up[k][u][0]
        if u == v:
            return (u, max_e)
        for k in reversed(range(LOG)):
            if up[k][u][0] != -1 and up[k][u][0] != up[k][v][0]:
                max_e = max(max_e, max_edge[k][u], max_edge[k][v])
                u = up[k][u][0]
                v = up[k][v][0]
        max_e = max(max_e, up[0][u][1], up[0][v][1])
        return (up[0][u][0], max_e)

    def get_max_edge(u, v):
        ancestor, max_e = lca(u, v)
        return max_e

    A.sort()
    B.sort(key=lambda x: (depth[x], x))
    sum_ans = 0
    for a in A:
        b = B.pop()
        sum_ans += get_max_edge(a, b)
    print(sum_ans)

if __name__ == "__main__":
    main()
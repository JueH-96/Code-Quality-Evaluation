import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from math import log2, ceil
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        edges[A].append(B)
        edges[B].append(A)
    
    V = []
    for _ in range(K):
        V.append(int(data[index]))
        index += 1
    
    LOG = ceil(log2(N + 1)) if N > 1 else 1
    parent = [[-1] * (N + 1) for _ in range(LOG)]
    depth = [0] * (N + 1)
    
    def dfs(u, p):
        parent[0][u] = p
        for v in edges[u]:
            if v != p:
                depth[v] = depth[u] + 1
                dfs(v, u)
    
    dfs(1, -1)
    
    for k in range(1, LOG):
        for v in range(1, N + 1):
            if parent[k - 1][v] != -1:
                parent[k][v] = parent[k - 1][parent[k - 1][v]]
    
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        for k in range(LOG - 1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                u = parent[k][u]
        if u == v:
            return u
        for k in range(LOG - 1, -1, -1):
            if parent[k][u] != parent[k][v]:
                u = parent[k][u]
                v = parent[k][v]
        return parent[0][u]
    
    V_sorted = sorted(V, key=lambda x: depth[x])
    stack = []
    virtual_tree = set()
    stack.append(1)
    virtual_tree.add(1)
    
    for node in V_sorted:
        current = node
        while True:
            ancestor = lca(current, stack[-1])
            if ancestor == stack[-1]:
                break
            while depth[ancestor] < depth[stack[-1]]:
                popped = stack.pop()
                virtual_tree.add(popped)
            if ancestor not in virtual_tree:
                virtual_tree.add(ancestor)
            stack.append(ancestor)
            current = ancestor
        stack.append(node)
        virtual_tree.add(node)
    
    while stack:
        popped = stack.pop()
        virtual_tree.add(popped)
    
    print(len(virtual_tree))

if __name__ == "__main__":
    main()
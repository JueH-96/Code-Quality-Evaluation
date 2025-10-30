import sys

def main():
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    index = 0
    N = int(data[index])
    M = int(data[index+1])
    index += 2
    operations = []
    for _ in range(M):
        K_i = int(data[index])
        C_i = int(data[index+1])
        index += 2
        S_i = list(map(int, data[index:index+K_i]))
        index += K_i
        operations.append((C_i, S_i))
    # Sort operations by C_i ascending
    operations.sort(key=lambda x: x[0])
    # Initialize Union-Find
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    component_count = N
    total_cost = 0

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if size[root_u] < size[root_v]:
                parent[root_u] = root_v
                size[root_v] += size[root_u]
            else:
                parent[root_v] = root_u
                size[root_u] += size[root_v]
            return True
        return False

    for op in operations:
        C_i, S_i = op
        roots = set()
        for u in S_i:
            roots.add(find(u))
        roots = list(roots)
        if len(roots) >= 2:
            total_cost += (len(roots) - 1) * C_i
            target = roots[0]
            for root in roots[1:]:
                union(root, target)
            component_count -= (len(roots) - 1)
    
    if component_count == 1:
        print(total_cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()
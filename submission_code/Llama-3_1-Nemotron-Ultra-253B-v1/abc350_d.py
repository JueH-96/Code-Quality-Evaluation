import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    parent = list(range(N))
    size = [1] * N
    edge_count = [0] * N
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for _ in range(M):
        a = int(data[idx]) - 1
        idx += 1
        b = int(data[idx]) - 1
        idx += 1
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if size[root_a] < size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            size[root_a] += size[root_b]
            edge_count[root_a] += edge_count[root_b] + 1
        else:
            edge_count[root_a] += 1
    
    roots = set()
    for i in range(N):
        roots.add(find(i))
    
    ans = 0
    for root in roots:
        s = size[root]
        e = edge_count[root]
        ans += s * (s - 1) // 2 - e
    
    print(ans)

if __name__ == '__main__':
    main()
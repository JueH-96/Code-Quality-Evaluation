import sys

def main():
    sys.setrecursionlimit(1 << 25)
    lines = sys.stdin.read().splitlines()
    idx = 0
    N, M = map(int, lines[idx].split())
    idx += 1
    groups = []
    for _ in range(M):
        K, C = map(int, lines[idx].split())
        idx += 1
        A = list(map(int, lines[idx].split()))
        idx += 1
        groups.append((C, K, A))
    groups.sort()
    parent = list(range(N + 1))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if u_root < v_root:
            parent[v_root] = u_root
        else:
            parent[u_root] = v_root
    
    count = N
    total = 0
    for C, K, A in groups:
        roots = set()
        for a in A:
            roots.add(find(a))
        m = len(roots)
        if m >= 2:
            total += (m - 1) * C
            count -= (m - 1)
            roots = list(roots)
            main_root = roots[0]
            for root in roots[1:]:
                union(main_root, root)
    print(total if count == 1 else -1)

if __name__ == "__main__":
    main()
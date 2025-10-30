import sys
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    matches = []
    for _ in range(N-1):
        p, q = map(int, sys.stdin.readline().split())
        matches.append((p, q))
    
    max_nodes = 2 * N
    parent = list(range(max_nodes + 2))
    size = [0] * (max_nodes + 2)
    for i in range(1, N+1):
        size[i] = 1
    c = [0] * (max_nodes + 2)
    s = [None] * (max_nodes + 2)
    next_node = N + 1
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for p, q in matches:
        root_p = find(p)
        root_q = find(q)
        a = size[root_p]
        b = size[root_q]
        inv_ab = pow(a + b, MOD-2, MOD)
        c1 = a * inv_ab % MOD
        c2 = b * inv_ab % MOD
        new_node = next_node
        next_node += 1
        parent[new_node] = new_node
        size[new_node] = a + b
        parent[root_p] = new_node
        parent[root_q] = new_node
        c[root_p] = c1
        c[root_q] = c2
    
    def compute_s(x):
        if s[x] is not None:
            return s[x]
        path = []
        current = x
        while True:
            path.append(current)
            next_p = parent[current]
            if next_p == current:
                break
            current = next_p
        s_val = 0
        s[current] = s_val
        for node in reversed(path):
            parent_node = parent[node]
            s_val = (c[node] + s[parent_node]) % MOD
            s[node] = s_val
        return s[x]
    
    result = []
    for i in range(1, N+1):
        compute_s(i)
        result.append(s[i])
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()
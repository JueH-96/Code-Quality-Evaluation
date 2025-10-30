MOD = 998244353

def mod_inv(a, mod=MOD):
    return pow(a, mod-2, mod)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    matches = []
    for i in range(N-1):
        p = int(data[1 + 2*i]) - 1
        q = int(data[2 + 2*i]) - 1
        matches.append((p, q))
    
    parent = [i for i in range(N)]
    size = [1] * N
    expect = [0] * N
    
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        u = find(u)
        v = find(v)
        if u == v:
            return
        if size[u] < size[v]:
            u, v = v, u
        parent[v] = u
        size[u] += size[v]
    
    for p, q in matches:
        u = find(p)
        v = find(v)
        a = size[u]
        b = size[v]
        total = a + b
        prob_u = (a * mod_inv(total)) % MOD
        prob_v = (b * mod_inv(total)) % MOD
        for x in [u, v]:
            if x == u:
                prob = prob_u
            else:
                prob = prob_v
            for member in range(N):
                if find(member) == x:
                    expect[member] = (expect[member] + prob) % MOD
        union(u, v)
    
    print(' '.join(map(str, expect)))

if __name__ == '__main__':
    main()
def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u - 1, v - 1))

    parent = list(range(n))
    sizes = [1] * n
    wins = [0] * n
    
    MOD = 998244353

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j, idx):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            a = sizes[root_i]
            b = sizes[root_j]
            
            prob_i = (a * pow(a + b, MOD - 2, MOD)) % MOD
            prob_j = (b * pow(a + b, MOD - 2, MOD)) % MOD
            
            wins[i] = (wins[i] + prob_i) % MOD
            wins[j] = (wins[j] + prob_j) % MOD

            if sizes[root_i] < sizes[root_j]:
                parent[root_i] = root_j
                sizes[root_j] += sizes[root_i]
            else:
                parent[root_j] = root_i
                sizes[root_i] += sizes[root_j]

    for i, (u, v) in enumerate(edges):
        union(u, v, i)

    print(*wins)

solve()
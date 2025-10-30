MOD = 998244353

class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
        self.members = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            self.members[i].append(i)
    
    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]  # Path compression
            u = self.parent[u]
        return u
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return
        # Union by size
        if self.size[root_u] < self.size[root_v]:
            root_u, root_v = root_v, root_u
        # Merge smaller into larger
        self.parent[root_v] = root_u
        self.size[root_u] += self.size[root_v]
        self.members[root_u].extend(self.members[root_v])

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    matches = []
    index = 1
    for _ in range(n - 1):
        p = int(data[index])
        q = int(data[index + 1])
        matches.append((p, q))
        index += 2
    
    dsu = DSU(n)
    E = [0] * (n + 1)
    
    for p, q in matches:
        root_p = dsu.find(p)
        root_q = dsu.find(q)
        a = dsu.size[root_p]
        b = dsu.size[root_q]
        total = (a + b) % MOD
        if total == 0:
            continue  # This case should not occur as per problem constraints
        inv_total = pow(total, MOD - 2, MOD)
        contrib_p = (a * inv_total) % MOD
        contrib_q = (b * inv_total) % MOD
        
        # Add contribution to all members of root_p's team
        for x in dsu.members[root_p]:
            E[x] = (E[x] + contrib_p) % MOD
        
        # Add contribution to all members of root_q's team
        for x in dsu.members[root_q]:
            E[x] = (E[x] + contrib_q) % MOD
        
        # Merge the two teams
        dsu.union(p, q)
    
    # Prepare the output
    output = ' '.join(map(str, E[1:n+1]))
    print(output)

if __name__ == '__main__':
    main()
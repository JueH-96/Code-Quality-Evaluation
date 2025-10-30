class DSU:
    def __init__(self, size):
        self.n = size
        self.parent = list(range(size))
        self.parity = [0] * size
        self.rank = [1] * size

    def find(self, u):
        path = []
        while self.parent[u] != u:
            path.append(u)
            u = self.parent[u]
        # Compute cumulative parity
        cumulative = [0] * (len(path) + 1)
        for i in range(len(path) - 1, -1, -1):
            cumulative[i] = self.parity[path[i]] ^ cumulative[i + 1]
        # Update path nodes to point directly to root
        for i in range(len(path)):
            self.parent[path[i]] = u
            self.parity[path[i]] = cumulative[i]
        return (u, cumulative[0] if path else 0)

    def unite(self, u, v, c):
        root_u, p_u = self.find(u)
        root_v, p_v = self.find(v)
        if root_u == root_v:
            return (p_u ^ p_v) == c
        # Union by rank
        if self.rank[root_u] < self.rank[root_v]:
            root_u, root_v = root_v, root_u
            p_u, p_v = p_v, p_u
            u, v = v, u
        self.parent[root_v] = root_u
        self.parity[root_v] = p_u ^ c ^ p_v
        if self.rank[root_u] == self.rank[root_v]:
            self.rank[root_u] += 1
        return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    dsu = DSU(2 * N)
    contradiction = False
    for _ in range(M):
        A = int(data[idx])
        idx +=1
        B = int(data[idx])
        idx +=1
        C = int(data[idx])
        idx +=1
        x_A = 2 * (A - 1)
        h_B = 2 * (B - 1) + 1
        if not dsu.unite(x_A, h_B, C):
            contradiction = True
            break
    if contradiction:
        print(-1)
    else:
        roots_val = [-1] * (2 * N)
        values = [-1] * (2 * N)
        for i in range(2 * N):
            root, parity = dsu.find(i)
            if roots_val[root] == -1:
                roots_val[root] = 0
            values[i] = roots_val[root] ^ parity
        result = []
        for i in range(1, N + 1):
            x_i = values[2 * (i - 1)]
            h_i = values[2 * (i - 1) + 1]
            result.append(str(x_i ^ h_i))
        print(''.join(result))

if __name__ == '__main__':
    main()
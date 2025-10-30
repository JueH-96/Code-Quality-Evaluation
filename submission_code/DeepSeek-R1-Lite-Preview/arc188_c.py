class UnionFindXOR:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.rank = [0] * (size + 1)
        self.xor_to_parent = [0] * (size + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.xor_to_parent[x] ^= self.xor_to_parent[original_parent]
        return self.parent[x]
    
    def union(self, x, y, b):
        xr = self.find(x)
        yr = self.find(y)
        if xr != yr:
            if self.rank[xr] < self.rank[yr]:
                self.parent[xr] = yr
                self.xor_to_parent[xr] = self.xor_to_parent[x] ^ self.xor_to_parent[y] ^ b
            else:
                self.parent[yr] = xr
                self.xor_to_parent[yr] = self.xor_to_parent[x] ^ self.xor_to_parent[y] ^ b
                if self.rank[xr] == self.rank[yr]:
                    self.rank[xr] += 1
        else:
            if self.xor_to_parent[x] ^ self.xor_to_parent[y] != b:
                return False
        return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    testimonies = []
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        testimonies.append((A, B, C))
        index += 3
    
    uf = UnionFindXOR(N)
    from collections import defaultdict
    A_to_testimonies = defaultdict(list)
    for A, B, C in testimonies:
        A_to_testimonies[A].append((B, C))
    
    for A, lst in A_to_testimonies.items():
        if len(lst) >= 2:
            B1, C1 = lst[0]
            for i in range(1, len(lst)):
                B2, C2 = lst[i]
                if not uf.union(B1, B2, C1 ^ C2):
                    print(-1)
                    return
    
    X = [0] * (N + 1)
    H = [0] * (N + 1)
    assigned = [False] * (N + 1)
    def dfs(u, val):
        if assigned[u]:
            if H[u] != val:
                return False
            return True
        H[u] = val
        assigned[u] = True
        for v in graph.get(u, []):
            val_v = H[u] ^ xor_edge[(u, v)]
            if not dfs(v, val_v):
                return False
        return True
    
    graph = defaultdict(list)
    xor_edge = {}
    for A, B, C in testimonies:
        graph[A].append(B)
        xor_edge[(A, B)] = C
    
    for i in range(1, N + 1):
        if not assigned[i]:
            if not dfs(i, 0):
                print(-1)
                return
    
    for A, B, C in testimonies:
        X[A] = H[A] ^ H[B] ^ C
    
    result = ''.join(str(X[i]) for i in range(1, N + 1))
    print(result)

if __name__ == '__main__':
    main()
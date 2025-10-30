def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    operations = []
    index = 2
    for _ in range(Q):
        L = int(data[index])
        R = int(data[index + 1])
        C = int(data[index + 2])
        operations.append((C, L - 1, R - 1))
        index += 3
    
    operations.sort()
    
    parent = [i for i in range(N + Q)]
    
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root
            return True
        return False
    
    class SegmentTree:
        def __init__(self, size):
            self.N = 1
            while self.N < size:
                self.N <<= 1
            self.parent = [i for i in range(2 * self.N)]
        
        def find(self, x):
            return self._find(x + self.N)
        
        def union_range(self, l, r, value):
            l += self.N
            r += self.N + 1
            while l < r:
                if l % 2 == 1:
                    self.parent[l] = value
                    l += 1
                if r % 2 == 1:
                    r -= 1
                    self.parent[r] = value
                l >>= 1
                r >>= 1
        
        def _find(self, x):
            while x >= self.N:
                if self.parent[x] == x - self.N:
                    break
                x = self.parent[x]
            return x - self.N
    
    st = SegmentTree(N)
    
    total_cost = 0
    connected_components = N + Q
    
    for cost, L, R in operations:
        hub = N + Q - 1 - (Q - 1)
        hub = N + Q - 1
        # Find representative of the hub
        hub_root = find(hub)
        # Find representative of the range [L, R]
        # To find the root of the range, we need to find the root of any vertex in [L, R]
        if L <= R:
            v = L
            v_root = find(st.find(v))
            if v_root != hub_root:
                parent[hub_root] = v_root
                total_cost += cost
                connected_components -= 1
            # Union the range [L, R] into one component
            st.union_range(L, R, v_root)
    
    if connected_components == 1:
        print(total_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1

    operations = []
    for i in range(1, Q+1):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        operations.append((L, R, C))

    # Check if all nodes are covered
    delta = [0] * (N + 2)
    for op in operations:
        L, R, C = op
        delta[L] += 1
        delta[R + 1] -= 1

    covered = [0] * (N + 1)  # 1-based
    for i in range(1, N+1):
        covered[i] = covered[i-1] + delta[i]

    for i in range(1, N+1):
        if covered[i] == 0:
            print(-1)
            return

    # For each node, find the minimal C and corresponding operation
    min_C = [float('inf')] * (N + 1)  # 1-based
    min_op = [-1] * (N + 1)  # stores the operation index (1-based)

    for j in range(1, N+1):
        for i in range(Q):
            L, R, C = operations[i]
            if L <= j <= R:
                if C < min_C[j]:
                    min_C[j] = C
                    min_op[j] = i + 1  # operations are 1-based in this context

    # Check if any node has no minimal operation
    for j in range(1, N+1):
        if min_op[j] == -1:
            print(-1)
            return

    # Count how many nodes chose each operation
    count = [0] * (Q + 1)  # operations are 1-based
    for j in range(1, N+1):
        op_idx = min_op[j]
        count[op_idx] += 1

    # Check if any operation has zero count
    for i in range(1, Q+1):
        if count[i] == 0:
            print(-1)
            return

    # Prepare DSU
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size + 1))  # 1-based indexing
            self.rank = [1] * (size + 1)
        
        def find(self, u):
            while self.parent[u] != u:
                self.parent[u] = self.parent[self.parent[u]]
                u = self.parent[u]
            return u
        
        def union(self, u, v):
            u_root = self.find(u)
            v_root = self.find(v)
            if u_root == v_root:
                return
            if self.rank[u_root] < self.rank[v_root]:
                self.parent[u_root] = v_root
                self.rank[v_root] += self.rank[u_root]
            else:
                self.parent[v_root] = u_root
                self.rank[u_root] += self.rank[v_root]

    total_nodes = N + Q
    dsu = DSU(total_nodes)

    # Connect each node to its minimal hub
    for j in range(1, N+1):
        op_idx = min_op[j]
        hub = N + op_idx
        dsu.union(j, hub)

    # Check if all nodes and hubs are connected
    root = dsu.find(1)
    for j in range(1, N+1):
        if dsu.find(j) != root:
            print(-1)
            return

    for hub in range(N+1, N+Q+1):
        if dsu.find(hub) != root:
            print(-1)
            return

    # Sum the minimal C's
    total = sum(min_C[1:N+1])
    print(total)

if __name__ == '__main__':
    main()
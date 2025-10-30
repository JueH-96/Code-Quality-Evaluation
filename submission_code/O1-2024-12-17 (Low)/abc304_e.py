def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # If performance in reading is critical, we can use sys.stdin directly.
    # But this should suffice for most environments.

    # DSU (Disjoint Set Union) / Union-Find implementation
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.size = [1]*n
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                if self.size[rx] < self.size[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                self.size[rx] += self.size[ry]
    
    idx = 0
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1

    dsu = DSU(N)

    # Read the edges of G and unify them in DSU
    for _ in range(M):
        u = int(input_data[idx]); idx += 1
        v = int(input_data[idx]); idx += 1
        # Zero-based
        dsu.union(u-1, v-1)
    
    K = int(input_data[idx]); idx += 1

    # We will record the pairs of DSU representatives that must never merge
    # i.e. if rep_x and rep_y are in the same set upon adding a new edge => "No"
    # We'll store them in a dict: forbid[rep_x] = set of rep_y that are forbidden
    # and likewise forbid[rep_y] = set of rep_x.
    forbid = {}
    def add_forbid(a, b):
        if a not in forbid:
            forbid[a] = set()
        forbid[a].add(b)
    
    # Read the K "no path" constraints
    # We already know they are in different components in G, so record them in forbid
    for _ in range(K):
        x = int(input_data[idx]); idx += 1
        y = int(input_data[idx]); idx += 1
        rx = dsu.find(x-1)
        ry = dsu.find(y-1)
        if rx == ry:
            # The problem statement says G is already good, so this should not happen.
            # We'll ignore or pass silently.
            pass
        else:
            add_forbid(rx, ry)
            add_forbid(ry, rx)
    
    Q = int(input_data[idx]); idx += 1

    out = []
    for _ in range(Q):
        p = int(input_data[idx]); idx += 1
        q = int(input_data[idx]); idx += 1
        rp = dsu.find(p-1)
        rq = dsu.find(q-1)
        if rp == rq:
            # Already in the same DSU set, so adding this edge changes nothing => still good
            out.append("Yes")
        else:
            # If rp and rq are in each other's forbidden sets => No
            # else => Yes
            if rp in forbid and rq in forbid[rp]:
                out.append("No")
            else:
                out.append("Yes")
    
    print("
".join(out))

# Do not forget to call main().
if __name__ == "__main__":
    main()
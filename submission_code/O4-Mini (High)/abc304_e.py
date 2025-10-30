import sys
def main():
    import sys
    input = sys.stdin.readline

    # Read number of vertices N and edges M
    N, M = map(int, input().split())
    
    # DSU (Disjoint Set Union) / Union-Find data structure
    class DSU:
        __slots__ = ("p", "r")
        def __init__(self, n):
            self.p = list(range(n))
            self.r = [0] * n
        def find(self, x):
            # iterative with path compression
            p = self.p
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
        def union(self, a, b):
            a = self.find(a)
            b = self.find(b)
            if a == b:
                return False
            # union by rank
            if self.r[a] < self.r[b]:
                a, b = b, a
            self.p[b] = a
            if self.r[a] == self.r[b]:
                self.r[a] += 1
            return True

    dsu = DSU(N)
    # Read and union the M edges
    for _ in range(M):
        u, v = map(int, input().split())
        dsu.union(u-1, v-1)

    # Read K forbidden pairs (x_i, y_i)
    K = int(input())
    forbidden = set()
    find = dsu.find
    # We'll encode an unordered pair of components (a,b) with a<b as key = a*N + b
    # so as to do O(1) membership checks.
    NN = N
    for _ in range(K):
        x, y = map(int, input().split())
        a = find(x-1)
        b = find(y-1)
        if a != b:
            if a > b:
                a, b = b, a
            forbidden.add(a * NN + b)

    # Process Q queries
    Q = int(input())
    out = []
    out_append = out.append
    contains = forbidden.__contains__
    for _ in range(Q):
        p, q = map(int, input().split())
        a = find(p-1)
        b = find(q-1)
        if a == b:
            # the new edge is inside one component -> no change in connectivity
            out_append("Yes")
        else:
            if a > b:
                a, b = b, a
            key = a * NN + b
            if contains(key):
                out_append("No")
            else:
                out_append("Yes")

    # Print answers
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
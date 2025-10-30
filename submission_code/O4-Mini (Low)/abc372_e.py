import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())

    # Disjoint Set Union with union by size and path compression
    parent = list(range(N+1))
    size = [1] * (N+1)

    # For each root, store a list of up to 10 largest vertices in its component (sorted descending)
    comp = [[i] for i in range(N+1)]
    MAXK = 10

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # make ra the larger component
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        # merge the top lists
        la = comp[ra]
        lb = comp[rb]
        # two-pointer merge on descending lists, but only need up to MAXK
        merged = []
        i = j = 0
        while len(merged) < MAXK and i < len(la) and j < len(lb):
            if la[i] >= lb[j]:
                merged.append(la[i])
                i += 1
            else:
                merged.append(lb[j])
                j += 1
        # fill from whichever remains
        while len(merged) < MAXK and i < len(la):
            merged.append(la[i])
            i += 1
        while len(merged) < MAXK and j < len(lb):
            merged.append(lb[j])
            j += 1
        comp[ra] = merged
        # optional: free memory
        comp[rb] = []

    out = []
    for _ in range(Q):
        line = input().split()
        t = int(line[0])
        if t == 1:
            u = int(line[1]); v = int(line[2])
            union(u, v)
        else:
            v = int(line[1]); k = int(line[2])
            rv = find(v)
            lst = comp[rv]
            if len(lst) >= k:
                out.append(str(lst[k-1]))
            else:
                out.append("-1")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
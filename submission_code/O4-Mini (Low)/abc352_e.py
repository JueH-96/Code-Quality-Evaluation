import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0])
    M = int(line[1])
    ops = []
    # Read operations
    for _ in range(M):
        ki_ci = data.readline().split()
        ki = int(ki_ci[0])
        ci = int(ki_ci[1])
        verts = list(map(int, data.readline().split()))
        ops.append((ci, verts))
    # Sort operations by cost ascending
    ops.sort(key=lambda x: x[0])
    # DSU
    parent = list(range(N+1))
    rank = [0]*(N+1)
    def find(x):
        while parent[x]!=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra = find(a)
        rb = find(b)
        if ra==rb:
            return False
        if rank[ra]<rank[rb]:
            parent[ra] = rb
        else:
            parent[rb] = ra
            if rank[ra]==rank[rb]:
                rank[ra]+=1
        return True
    total = 0
    comps = N
    # Process each hyperedge (complete subgraph) in increasing cost
    for cost, verts in ops:
        # pick first vertex as representative
        v0 = verts[0]
        r0 = find(v0)
        for v in verts[1:]:
            rv = find(v)
            if rv!=r0:
                # connect v to v0
                union(r0, rv)
                total += cost
                comps -= 1
                # update r0 to the new root of the merged component
                r0 = find(r0)
                if comps == 1:
                    # already fully connected; can exit early
                    break
        if comps == 1:
            break
    # If not connected, print -1
    if comps != 1:
        print(-1)
    else:
        print(total)

if __name__ == "__main__":
    main()
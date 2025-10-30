import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # DSU arrays
    parent = [i for i in range(N+2)]
    sz = [1] * (N+2)
    minpos = [i for i in range(N+2)]
    maxpos = [i for i in range(N+2)]
    col = [i for i in range(N+2)]
    # count of cells per color
    count = [0] * (N+2)
    for i in range(1, N+1):
        count[i] = 1

    # DSU find with path compression
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # DSU union by size, returns new root
    def union(u, v):
        ru = find(u)
        rv = find(v)
        if ru == rv:
            return ru
        # both ru and rv have same color when we call union
        if sz[ru] < sz[rv]:
            ru, rv = rv, ru
        # attach rv under ru
        parent[rv] = ru
        sz[ru] += sz[rv]
        # update segment bounds
        if minpos[rv] < minpos[ru]:
            minpos[ru] = minpos[rv]
        if maxpos[rv] > maxpos[ru]:
            maxpos[ru] = maxpos[rv]
        # col[ru] remains as they are same
        return ru

    out = []
    for _ in range(Q):
        data = input().split()
        t = int(data[0])
        if t == 1:
            x = int(data[1])
            c = int(data[2])
            # find the set of x
            r = find(x)
            oldc = col[r]
            if oldc == c:
                continue
            # recolor this set
            group_size = sz[r]
            count[oldc] -= group_size
            count[c] += group_size
            col[r] = c
            # try merging with left neighbor
            # recompute root of x
            r = find(x)
            lpos = minpos[r]
            if lpos > 1:
                nb = find(lpos - 1)
                if nb != r and col[nb] == c:
                    # merge and no count change
                    r = union(r, nb)
                    # ensure color stays c
                    col[r] = c
            # try merging with right neighbor
            r = find(x)
            rpos = maxpos[r]
            if rpos < N:
                nb = find(rpos + 1)
                if nb != r and col[nb] == c:
                    r = union(r, nb)
                    col[r] = c

        else:
            # query type 2
            c = int(data[1])
            out.append(str(count[c]))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
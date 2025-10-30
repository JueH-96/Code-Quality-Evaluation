import sys
import threading
def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        u,v,w = map(int, input().split())
        edges.append((w, u-1, v-1))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    # zero‐based vertices
    # count occurrences in A and B at each vertex
    cntA = [0]*N
    cntB = [0]*N
    for x in A:
        cntA[x-1] += 1
    for x in B:
        cntB[x-1] += 1

    # DSU
    parent = list(range(N))
    # we don't need rank heuristics too much, but we can
    size = [1]*N

    def find(x):
        while parent[x]!=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    # sort edges by weight ascending
    edges.sort()
    for w,u,v in edges:
        ru = find(u)
        rv = find(v)
        if ru==rv:
            continue
        # match A in ru with B in rv
        m1 = min(cntA[ru], cntB[rv])
        # match A in rv with B in ru
        m2 = min(cntA[rv], cntB[ru])
        # add to total
        total += w*(m1 + m2)
        # subtract matched
        cntA[ru] -= m1
        cntB[rv] -= m1
        cntA[rv] -= m2
        cntB[ru] -= m2
        # union by size
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        # now ru is bigger
        parent[rv] = ru
        size[ru] += size[rv]
        # accumulate leftover counts
        cntA[ru] += cntA[rv]
        cntB[ru] += cntB[rv]
    print(total)

if __name__ == "__main__":
    main()
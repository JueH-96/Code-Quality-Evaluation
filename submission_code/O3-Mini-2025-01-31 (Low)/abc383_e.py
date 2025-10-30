def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # Read inputs
    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((w, u-1, v-1))
    # Reading sequences A and B (convert to 0-index)
    A_list = list(map(lambda x: int(x)-1, input().split()))
    B_list = list(map(lambda x: int(x)-1, input().split()))

    # Build count arrays for A and B (vertices may appear multiple times)
    countA = [0]*N
    countB = [0]*N
    for a in A_list:
        countA[a] += 1
    for b in B_list:
        countB[b] += 1

    # sort edges by ascending weight (this is effectively Kruskal's process)
    edges.sort(key=lambda x: x[0])

    # DSU with extra info: for each component, keep:
    #   cntA: total count of A labels,
    #   cntB: total count of B labels,
    #   paired: number of pairs already matched within this component.
    parent = list(range(N))
    size = [1] * N
    cntA = countA[:]  # copy initial counts (per vertex)
    cntB = countB[:]
    paired = [0] * N  # initially, no pair is internally matched

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ans = 0

    # DSU union routine that merges the two components and
    # returns the number of new pairs unlocked multiplied by cost.
    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        # Ensure ru is the larger component
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        # new counts after union:
        newA = cntA[ru] + cntA[rv]
        newB = cntB[ru] + cntB[rv]
        # maximum pairs possible in the union:
        newPairs = min(newA, newB)
        # Additional pairs that only become matchable by crossing this edge:
        addPairs = newPairs - (paired[ru] + paired[rv])
        ans += w * addPairs
        # update the component information
        cntA[ru] = newA
        cntB[ru] = newB
        paired[ru] = newPairs

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()
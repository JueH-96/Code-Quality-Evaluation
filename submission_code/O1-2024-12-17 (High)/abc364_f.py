def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    # Fast parse
    N, Q = map(int, input_data[:2])
    LRCA = input_data[2:]
    
    # -------------------------------------------------------------
    # 1) Read intervals and perform a coverage check via difference array
    #    If any original node is uncovered, answer is -1 immediately.
    # -------------------------------------------------------------
    diff = [0]*(N+1)  # difference array for coverage
    intervals = []
    idx = 0
    for i in range(Q):
        L = int(LRCA[idx]); R = int(LRCA[idx+1]); C = int(LRCA[idx+2])
        idx += 3
        # Convert to 0-based for convenience
        L -= 1
        R -= 1
        intervals.append((L, R, C, i))  # store original index too
        diff[L] += 1
        if R+1 < N:
            diff[R+1] -= 1

    coverage = [0]*N
    cur = 0
    for i in range(N):
        cur += diff[i]
        coverage[i] = cur

    # If any position is uncovered => disconnected => -1
    for i in range(N):
        if coverage[i] == 0:
            print(-1)
            return

    # -------------------------------------------------------------
    # 2) Sort intervals by cost ascending
    # -------------------------------------------------------------
    intervals.sort(key=lambda x: x[2])  # sort by cost

    # -------------------------------------------------------------
    # 3) DSU (Disjoint Set) with small-to-large merging.
    #    We'll keep track of "members" for each leader, so we can
    #    redirect skip[] pointers en masse when two sets unify.
    # -------------------------------------------------------------
    class DSU:
        __slots__ = ('parent','size','members')

        def __init__(self, n):
            self.parent = list(range(n))
            self.size = [1]*n
            # members[v] = list of all nodes in the connected component whose leader is v
            self.members = [[] for _ in range(n)]
            for i in range(n):
                self.members[i].append(i)
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def unionLarge(self, x, y, skipTarget, skipArr):
            # Unifies the sets of x,y, and for the smaller set's members,
            # set skip[...] = skipTarget
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            if self.size[rx] < self.size[ry]:
                rx, ry = ry, rx
            # Now rx is the bigger leader
            self.parent[ry] = rx
            # Move members of ry into rx
            for node in self.members[ry]:
                self.members[rx].append(node)
                skipArr[node] = skipTarget
            self.members[ry].clear()
            self.size[rx] += self.size[ry]
            return True

    # We have N original nodes (indexed 0..N-1) and Q super-nodes (indexed N..N+Q-1).
    dsu = DSU(N+Q)

    # -------------------------------------------------------------
    # 4) skip-array so we can skip over nodes already connected
    #    to the same super-node in the Kruskal-like process.
    #    skip[i] gives the "next" node index to check after i.
    #    For i in [0..N-2], skip[i] = i+1; skip[N-1] = N (sentinel).
    # -------------------------------------------------------------
    skip = list(range(N+1))
    for i in range(N):
        skip[i] = i+1
    skip[N] = N  # sentinel

    # Helper to get the next node not yet in the same set as sLeader
    def getNextDistinct(sLeader, p):
        # sLeader is the leader of the super-node in DSU.
        # p is an index among [0..N], p==N means "out of range".
        while p < N and dsu.find(p) == sLeader:
            p = skip[p]
        return p

    # -------------------------------------------------------------
    # 5) Kruskal-like process:
    #    - Process intervals in ascending cost
    #    - For each interval i with cost c, unify super-node (N+i_idx) with
    #      all distinct DSU-components in [L..R].
    #    - Use skip[] to jump over nodes already connected.
    #    - Use unionLarge to small-to-large merge sets and set skip[] en masse.
    # -------------------------------------------------------------
    merges = 0
    total_cost = 0
    # Total number of vertices = N + Q, so MST will have (N+Q-1) edges if connected
    MST_EDGES = N + Q - 1

    for (l, r, c, idx) in intervals:
        # Super-node index in DSU
        sNode = N + idx
        sNodeLeader = dsu.find(sNode)
        p = l
        while True:
            # Find the next candidate in [l..r] that is NOT in sNode's component
            # If p > r or p == N, we are done with this interval
            p = getNextDistinct(sNodeLeader, p)
            if p > r or p == N:
                break
            # Attempt to unify
            oldLeaderP = dsu.find(p)
            skipTarget = skip[p]  # We'll jump to skipTarget after merging
            if dsu.unionLarge(sNode, p, skipTarget, skip):
                total_cost += c
                merges += 1
                if merges == MST_EDGES:
                    print(total_cost)
                    return
            # Move p to skipTarget
            p = skipTarget
        # Update sNodeLeader after any merges
        sNodeLeader = dsu.find(sNode)

    # -------------------------------------------------------------
    # 6) If we didn't get enough merges, the graph is disconnected
    # -------------------------------------------------------------
    if merges < MST_EDGES:
        print(-1)
    else:
        print(total_cost)
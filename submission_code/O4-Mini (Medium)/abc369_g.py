import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v,w = map(int, sys.stdin.readline().split())
        adj[u].append((v,w))
        adj[v].append((u,w))
    parent = [0]*(N+1)
    w2p = [0]*(N+1)
    depth = [0]*(N+1)
    tin = [0]*(N+1)
    tout = [0]*(N+1)
    # iterative DFS for Euler tour, compute parent, depth, tin/tout, leaves
    time = 1
    leaves = []
    stack = [(1, 0, False)]
    while stack:
        node, par, closing = stack.pop()
        if not closing:
            parent[node] = par
            tin[node] = time
            time += 1
            # gather children
            kids = []
            for v,w in adj[node]:
                if v == par: continue
                kids.append((v,w))
                # set child depth and parent info immediately
                depth[v] = depth[node] + w
                parent[v] = node
                w2p[v] = w
            # push closing marker
            stack.append((node, par, True))
            if not kids:
                # no children => a leaf (excluding root if no kids)
                leaves.append(node)
            # push children in reverse so first child is processed first
            for v,w in reversed(kids):
                stack.append((v, node, False))
        else:
            # closing
            tout[node] = time-1
    # BIT for range-add, point-query
    class BIT:
        def __init__(self,n):
            self.n = n
            self.bit = [0]*(n+5)
        def add(self,i,v):
            # add v at position i
            n = self.n
            while i <= n:
                self.bit[i] += v
                i += i & -i
        def prefix_sum(self,i):
            s = 0
            while i>0:
                s += self.bit[i]
                i -= i & -i
            return s
        # range add [l..r] by v
        def range_add(self,l,r,v):
            if l>r: return
            self.add(l, v)
            self.add(r+1, -v)
        # point query at i
        def point_query(self,i):
            return self.prefix_sum(i)
    bit = BIT(N+2)
    # covered edges marker: covered_edge[v]=True if edge parent->v is covered
    covered_edge = [False]*(N+1)
    # max-heap of leaves by marginal = depth - covered
    import heapq
    pq = []
    for leaf in leaves:
        # initial covered is 0
        heapq.heappush(pq, (-depth[leaf], leaf))
    res = [0]*(N+1)
    # build answers for K=1..N
    for k in range(1, N+1):
        m = 0
        # get best marginal
        while pq:
            neg_old, leaf = heapq.heappop(pq)
            old = -neg_old
            cur_cov = bit.point_query(tin[leaf])
            new = depth[leaf] - cur_cov
            if new != old:
                # stale, re-push updated
                heapq.heappush(pq, (-new, leaf))
                continue
            m = new
            break
        # if pq empty, m stays 0
        res[k] = res[k-1] + m
        if m > 0:
            # cover edges on the path from leaf up until already covered
            cur = leaf
            while cur != 1 and not covered_edge[cur]:
                # cover edge parent[cur] -> cur
                covered_edge[cur] = True
                w = w2p[cur]
                # range add w to subtree of cur
                bit.range_add(tin[cur], tout[cur], w)
                cur = parent[cur]
        # else m==0: no more coverage possible
    # print answers: 2 * res[k]
    out = sys.stdout
    for k in range(1, N+1):
        out.write(str(2*res[k]))
        out.write("
")

if __name__ == "__main__":
    main()
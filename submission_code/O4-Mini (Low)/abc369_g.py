import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v,l = map(int, sys.stdin.readline().split())
        adj[u].append((v,l))
        adj[v].append((u,l))
    # BFS from root=1: get parent, weight to parent, dist
    from collections import deque
    parent = [0]*(N+1)
    wpar = [0]*(N+1)
    dist = [0]*(N+1)
    deg = [0]*(N+1)
    for u in range(1,N+1):
        deg[u] = len(adj[u])
    q = deque([1])
    parent[1] = 1
    dist[1] = 0
    while q:
        u = q.popleft()
        for v,l in adj[u]:
            if v==parent[u]: continue
            parent[v] = u
            wpar[v] = l
            dist[v] = dist[u] + l
            q.append(v)
    # DSU for skipping covered edges
    fa = list(range(N+1))
    def find(x):
        while fa[x]!=x:
            fa[x] = fa[fa[x]]
            x = fa[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb:
            fa[ra] = rb
    # identify leaves (exclude root)
    leaves = []
    for u in range(2, N+1):
        if deg[u]==1:
            leaves.append(u)
    # max-heap of (gain, node)
    import heapq
    heap = []
    for v in leaves:
        # initial gain = full dist
        heapq.heappush(heap, (-dist[v], v))
    ans = [0]*(N+1)
    total = 0
    picks = 0
    maxpicks = len(leaves)
    # function to compute real gain and cover edges
    def get_gain(v):
        g = 0
        cur = v
        # move up until root
        while True:
            r = find(cur)
            if r==1: break
            # r is node whose edge to parent not yet covered
            g += wpar[r]
            # cover that edge by union r with parent[r]
            union(r, parent[r])
        return g
    # greedy pick
    for k in range(1, N+1):
        if picks < maxpicks:
            # find next real best leaf
            while True:
                if not heap: break
                negg, v = heapq.heappop(heap)
                guessed = -negg
                real = get_gain(v)
                if real < guessed:
                    # some overlap was covered, re-insert with updated gain
                    heapq.heappush(heap, (-real, v))
                    continue
                # accept
                total += real
                picks += 1
                break
        # total sum of gains is sum of uncovered paths; walk length = 2*total
        ans[k] = total * 2
    # print answers
    out = sys.stdout
    for k in range(1, N+1):
        out.write(str(ans[k]) + "
")

if __name__ == "__main__":
    main()
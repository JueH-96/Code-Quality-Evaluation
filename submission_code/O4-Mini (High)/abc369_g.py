import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v, w = map(int, input().split())
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Build parent, depth, children, and post-order stack
    depth = [0] * (N+1)
    parent = [0] * (N+1)
    children = [[] for _ in range(N+1)]
    stack1 = [(1, 0, 0)]   # (node, parent, weight from parent)
    stack2 = []            # to collect nodes in reverse post-order
    while stack1:
        v, p, w = stack1.pop()
        parent[v] = p
        # depth of root (p=0) stays 0
        if p != 0:
            depth[v] = depth[p] + w
        stack2.append(v)
        for to, wt in adj[v]:
            if to == p:
                continue
            children[v].append(to)
            stack1.append((to, v, wt))

    # Compute f[v] = max depth of any node in v's subtree,
    # heavy[v] = the child of v achieving that maximum
    f = [0] * (N+1)
    heavy = [0] * (N+1)
    # process in post-order so children are done before parent
    for v in reversed(stack2):
        # start with own depth
        best = depth[v]
        hb = 0
        for c in children[v]:
            if f[c] > best:
                best = f[c]
                hb = c
        f[v] = best
        heavy[v] = hb

    # Greedy: maintain a max-heap of available "trees" (roots),
    # keyed by the longest path from that root down to a leaf,
    # measured from the covered ancestor.
    import heapq
    H = []
    # initially only the whole tree at root=1, baseDepth=0, key = f[1]-0 = f[1]
    heapq.heappush(H, (-f[1], 1))

    cov = [0] * (N+1)
    total = 0
    # perform up to N picks
    for k in range(1, N+1):
        if H:
            negd, u = heapq.heappop(H)
            d = -negd
            total += d
            cov[k] = total
            # Traverse down the heavy path from u, pushing side branches
            cur = u
            while True:
                hc = heavy[cur]
                # for every child except the heavy child, push its subtree
                for c in children[cur]:
                    if c == hc:
                        continue
                    # key = f[c] - depth[cur]
                    keyc = f[c] - depth[cur]
                    heapq.heappush(H, (-keyc, c))
                if hc == 0:
                    break
                cur = hc
        else:
            # no more uncovered edges
            cov[k] = total

    # Output answers: for each K, answer = 2 * covered_weight
    out = sys.stdout
    for k in range(1, N+1):
        out.write(str(cov[k] * 2) + "
")

if __name__ == "__main__":
    main()
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    parents = list(map(int, input().split()))
    # build tree
    tree = [[] for _ in range(N+1)]
    for i, p in enumerate(parents, start=2):
        tree[p].append(i)
    # compute tin, tout, depth via DFS
    tin = [0] * (N+1)
    tout = [0] * (N+1)
    depth = [0] * (N+1)
    t = 0
    stack = [(1, 0, False, 0)]  # node, parent, visited_flag, depth
    while stack:
        u, p, visited, d = stack.pop()
        if not visited:
            depth[u] = d
            tin[u] = t
            t += 1
            stack.append((u, p, True, d))
            for v in tree[u][::-1]:
                stack.append((v, u, False, d+1))
        else:
            tout[u] = t
    # Prepare events by depth
    # events[d]: list of (l, r, delta)
    maxd = max(depth)
    events = [[] for _ in range(maxd+1)]
    # process queries
    for _ in range(M):
        x, y = map(int, input().split())
        dmin = depth[x]
        dmax = depth[x] + y
        if dmax > maxd:
            dmax = maxd
        if dmax < dmin:
            continue
        l = tin[x]
        r = tout[x] - 1
        # at depth dmax: +1
        events[dmax].append((l, r, 1))
        # at depth dmin-1: -1 (if >= 0)
        if dmin > 0:
            events[dmin-1].append((l, r, -1))
    # BIT for range add, point query
    size = N + 5
    BIT = [0] * (size)
    def bit_add(i, v):
        # add v at index i
        i += 1
        while i < size:
            BIT[i] += v
            i += i & -i
    def bit_sum(i):
        s = 0
        i += 1
        while i > 0:
            s += BIT[i]
            i -= i & -i
        return s
    def range_add(l, r, v):
        bit_add(l, v)
        bit_add(r+1, -v)
    # group nodes by depth
    nodes_at_depth = [[] for _ in range(maxd+1)]
    for u in range(1, N+1):
        nodes_at_depth[depth[u]].append(u)
    # sweep depths from maxd down to 0
    covered = 0
    for d in range(maxd, -1, -1):
        # apply events at depth d
        for l, r, delta in events[d]:
            range_add(l, r, delta)
        # check nodes at this depth
        for u in nodes_at_depth[d]:
            if bit_sum(tin[u]) > 0:
                covered += 1
    print(covered)

if __name__ == "__main__":
    main()
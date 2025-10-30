import sys
sys.setrecursionlimit(1000000)
def main():
    import sys
    input = sys.stdin.readline

    N, K = map(int, input().split())
    NK = N * K
    adj = [[] for _ in range(NK+1)]
    for _ in range(NK-1):
        u,v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # Trivial: K=1, each vertex is its own path
    if K == 1:
        print("Yes")
        return

    # Find one centroid (and possibly a second) of the tree
    size = [0]*(NK+1)
    parent = [0]*(NK+1)
    def dfs_sz(u,p):
        size[u] = 1
        parent[u] = p
        for v in adj[u]:
            if v == p: continue
            dfs_sz(v,u)
            size[u] += size[v]
    dfs_sz(1,0)
    # climb from 1 to centroid
    cent1 = 1
    p = 0
    while True:
        moved = False
        for v in adj[cent1]:
            if v == parent[cent1]: continue
            if size[v] > NK//2:
                cent1 = v
                moved = True
                break
        if not moved:
            break
    # maybe a second centroid
    cent2 = 0
    for v in adj[cent1]:
        if v == parent[cent1]: continue
        if size[v] == NK//2:
            cent2 = v
            break

    candidates = [cent1]
    if cent2 and cent2 != cent1:
        candidates.append(cent2)

    # Try each candidate root
    for root in candidates:
        paths_count = 0
        failed = False

        def dfs(u,p):
            nonlocal paths_count, failed
            used_here = 0
            # collect partial fragments from children
            frag_count = {}
            for v in adj[u]:
                if v == p: continue
                child = dfs(v, u)
                if failed:
                    return 0
                if child == 0:
                    # child's subtree completely partitioned, no fragment
                    continue
                # extend child's fragment by u
                ln = child + 1
                if ln == K:
                    used_here += 1
                else:
                    frag_count[ln] = frag_count.get(ln, 0) + 1
            # if two children each completed a path using u, that's impossible
            if used_here >= 2:
                failed = True
                return 0
            # if exactly one child+u made a full path, u is consumed, no other fragments allowed
            if used_here == 1:
                if any(cnt > 0 for cnt in frag_count.values()):
                    failed = True
                    return 0
                paths_count += 1
                return 0

            # otherwise we try to pair partials summing to K+1 (since u is double‐counted in each)
            # gather keys once
            keys = sorted(frag_count.keys())
            for ln in keys:
                cnt = frag_count.get(ln,0)
                if cnt <= 0:
                    continue
                target = K + 1 - ln
                if target not in frag_count:
                    continue
                if ln < target:
                    pairs = min(cnt, frag_count[target])
                elif ln == target:
                    pairs = cnt // 2
                else:
                    continue
                if pairs > 0:
                    frag_count[ln] -= pairs
                    frag_count[target] -= pairs
                    paths_count += pairs

            # now at most one fragment may remain to go up
            total_left = 0
            last_len = 0
            for ln, cnt in frag_count.items():
                if cnt > 0:
                    total_left += cnt
                    last_len = ln
                    if total_left > 1:
                        break
            if total_left > 1:
                failed = True
                return 0
            if total_left == 1:
                # propagate that one partial
                return last_len
            # no fragments at all: u itself starts a new fragment
            return 1

        top_ret = dfs(root, 0)
        # at top, we need no leftover fragment, and exactly N paths formed
        if not failed and top_ret == 0 and paths_count == N:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()
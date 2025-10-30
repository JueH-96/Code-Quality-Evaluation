def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # Parse input
    N, M = map(int, input_data[:2])
    p_list = list(map(int, input_data[2:2+(N-1)]))
    xy_list = input_data[2+(N-1):]

    # Build adjacency (children) from parent info
    # We know person i's parent is p_i (i >= 2)
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = p_list[i-2]
        children[p].append(i)

    # We will store queries (x_i, y_i)
    # Then for each query we compute c_i = depth[x_i] + y_i
    # We'll do an Euler Tour to flatten the tree in[] / out[] arrays.
    # Then we will do range max-updates over [in[x_i], out[x_i]) with c_i.
    # After all updates, for each node x, let best[x] = range_max(in[x]). If best[x] >= depth[x], x is covered.

    XY = []
    idx = 0
    for _ in range(M):
        x_i = int(xy_list[idx]); y_i = int(xy_list[idx+1])
        idx += 2
        XY.append((x_i, y_i))

    sys.setrecursionlimit(10**7)

    # Step 1: DFS to compute depth, in, out, eul-array
    depth = [0]*(N+1)
    in_pos = [0]*(N+1)
    out_pos = [0]*(N+1)
    eul = []
    current_time = 0

    def dfs(u, d):
        nonlocal current_time
        depth[u] = d
        in_pos[u] = current_time
        eul.append(u)
        current_time += 1
        for c in children[u]:
            dfs(c, d+1)
        out_pos[u] = current_time

    # The root is person 1 (given p_i <= i-1, so 1 is effectively the root)
    dfs(1, 0)

    # We have an array "best" that will store the segment-tree / fenwick-tree
    # approach: we need range-maximum updates. We'll do a lazy-segtree for range max.

    sizeST = 1
    while sizeST < N:
        sizeST <<= 1
    segtree = [-1]*(2*sizeST)
    lazy = [-1]*(2*sizeST)  # lazy propagation array for max

    def apply_set(idx, val):
        segtree[idx] = max(segtree[idx], val)
        lazy[idx] = max(lazy[idx], val)

    def push_down(idx):
        if lazy[idx] < 0:
            return
        for c in (idx*2, idx*2+1):
            apply_set(c, lazy[idx])
        lazy[idx] = -1

    def update_range(left, right, val, idx, start, end):
        if left >= end or right <= start:
            return
        if left <= start and end <= right:
            apply_set(idx, val)
            return
        push_down(idx)
        mid = (start+end)//2
        update_range(left, right, val, idx*2, start, mid)
        update_range(left, right, val, idx*2+1, mid, end)
        segtree[idx] = max(segtree[idx*2], segtree[idx*2+1])

    def update(left, right, val):
        # range update [left, right)
        update_range(left, right, val, 1, 0, sizeST)

    def get_value(pos, idx, start, end):
        if start+1 == end:
            return segtree[idx]
        push_down(idx)
        mid = (start+end)//2
        if pos < mid:
            return get_value(pos, idx*2, start, mid)
        else:
            return get_value(pos, idx*2+1, mid, end)

    def query(pos):
        return get_value(pos, 1, 0, sizeST)

    # Step 2: process each insurance
    # for (x,y), we set c = depth[x] + y
    # update [in[x], out[x]) with c
    for (x, y) in XY:
        c = depth[x] + y
        L = in_pos[x]
        R = out_pos[x]
        update(L, R, c)

    # Step 3: count how many nodes x have query(in[x]) >= depth[x]
    ans = 0
    for x in range(1, N+1):
        cmax = query(in_pos[x])
        if cmax >= depth[x]:
            ans += 1

    print(ans)

# Let's call solve() here as requested.
solve()
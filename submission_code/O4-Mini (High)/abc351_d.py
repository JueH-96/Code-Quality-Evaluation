def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    N = H * W

    # Flattened array marking magnet cells
    is_magnet = [False] * N
    for i in range(H):
        row = grid[i]
        base = i * W
        for j, c in enumerate(row):
            if c == '#':
                is_magnet[base + j] = True

    # is_good[idx] = True if cell is empty and has no adjacent magnet
    is_good = [False] * N
    for i in range(H):
        base = i * W
        for j in range(W):
            idx = base + j
            if not is_magnet[idx]:
                ok = True
                if i > 0 and is_magnet[idx - W]:
                    ok = False
                elif i < H - 1 and is_magnet[idx + W]:
                    ok = False
                elif j > 0 and is_magnet[idx - 1]:
                    ok = False
                elif j < W - 1 and is_magnet[idx + 1]:
                    ok = False
                if ok:
                    is_good[idx] = True

    # Find connected components among good cells
    comp_id = [-1] * N
    g_count = []   # size of each good‐component
    comp_idx = 0

    for start in range(N):
        if is_good[start] and comp_id[start] == -1:
            # BFS using a list as queue
            q = [start]
            comp_id[start] = comp_idx
            size = 1
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                ui = u // W
                uj = u - ui * W
                # four directions
                # up
                if ui > 0:
                    v = u - W
                    if is_good[v] and comp_id[v] == -1:
                        comp_id[v] = comp_idx
                        size += 1
                        q.append(v)
                # down
                if ui < H - 1:
                    v = u + W
                    if is_good[v] and comp_id[v] == -1:
                        comp_id[v] = comp_idx
                        size += 1
                        q.append(v)
                # left
                if uj > 0:
                    v = u - 1
                    if is_good[v] and comp_id[v] == -1:
                        comp_id[v] = comp_idx
                        size += 1
                        q.append(v)
                # right
                if uj < W - 1:
                    v = u + 1
                    if is_good[v] and comp_id[v] == -1:
                        comp_id[v] = comp_idx
                        size += 1
                        q.append(v)
            g_count.append(size)
            comp_idx += 1

    # If there are no good cells, every empty cell is "bad" => degree = 1
    if comp_idx == 0:
        print(1)
        return

    # For each good‐component, count how many distinct bad neighbors it has
    b_adj_count = [0] * comp_idx

    for i in range(H):
        base = i * W
        for j in range(W):
            idx = base + j
            # bad empty cell = empty but not good
            if not is_magnet[idx] and not is_good[idx]:
                seen = []
                # up
                if i > 0:
                    v = idx - W
                    if is_good[v]:
                        cid = comp_id[v]
                        if cid not in seen:
                            seen.append(cid)
                # down
                if i < H - 1:
                    v = idx + W
                    if is_good[v]:
                        cid = comp_id[v]
                        if cid not in seen:
                            seen.append(cid)
                # left
                if j > 0:
                    v = idx - 1
                    if is_good[v]:
                        cid = comp_id[v]
                        if cid not in seen:
                            seen.append(cid)
                # right
                if j < W - 1:
                    v = idx + 1
                    if is_good[v]:
                        cid = comp_id[v]
                        if cid not in seen:
                            seen.append(cid)
                for cid in seen:
                    b_adj_count[cid] += 1

    # Compute the maximum degree of freedom
    ans = 1
    for cid in range(comp_idx):
        val = g_count[cid] + b_adj_count[cid]
        if val > ans:
            ans = val

    print(ans)

if __name__ == "__main__":
    main()
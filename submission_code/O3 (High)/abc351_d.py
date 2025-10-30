import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1_500_000)
    read = sys.stdin.readline

    H, W = map(int, read().split())
    S = [read().strip() for _ in range(H)]
    N = H * W                         # total number of cells
    dirs = (-1, 1, 0, 0), (0, 0, -1, 1)

    # ------------------------------------------------------------------
    # 1. classify every cell
    #    empty  : '.'  (we can stay on it)
    #    magnet : '#'
    #    good   : empty cell with NO adjacent magnet
    #    bad    : empty cell WITH an adjacent magnet
    # ------------------------------------------------------------------
    is_empty = [False] * N
    is_good  = [False] * N

    for i in range(H):
        base = i * W
        for j, ch in enumerate(S[i]):
            if ch == '.':
                is_empty[base + j] = True

    di, dj = dirs
    for i in range(H):
        base = i * W
        for j in range(W):
            idx = base + j
            if not is_empty[idx]:
                continue          # a magnet
            good = True
            for k in range(4):
                ni, nj = i + di[k], j + dj[k]
                if 0 <= ni < H and 0 <= nj < W and S[ni][nj] == '#':
                    good = False
                    break
            if good:
                is_good[idx] = True

    # ------------------------------------------------------------------
    # 2. BFS on connected components consisting solely of good cells
    #    For every component:
    #        size            = number of good cells inside it
    #        adj_bad_count   = number of (distinct) bad cells adjacent to it
    #    freedom for any cell in that component = size + adj_bad_count
    # ------------------------------------------------------------------
    comp_id        = [-1] * N         # id for good cells, -1 = unvisited
    bad_seen_stamp = [-1] * N         # helper to count each bad cell once per component
    max_freedom    = 1                # at least every empty cell has freedom 1
    curr_comp_id   = 0
    q = deque()

    for i in range(H):
        base = i * W
        for j in range(W):
            idx = base + j
            if not is_good[idx] or comp_id[idx] != -1:
                continue

            comp_size, adj_bad = 0, 0
            comp_id[idx] = curr_comp_id
            q.append(idx)

            while q:
                v = q.popleft()
                comp_size += 1
                vi, vj = divmod(v, W)

                for k in range(4):
                    ni, nj = vi + di[k], vj + dj[k]
                    if not (0 <= ni < H and 0 <= nj < W):
                        continue
                    nidx = ni * W + nj
                    if not is_empty[nidx]:
                        continue                      # magnet, ignore
                    if is_good[nidx]:
                        if comp_id[nidx] == -1:
                            comp_id[nidx] = curr_comp_id
                            q.append(nidx)
                    else:                             # bad cell
                        if bad_seen_stamp[nidx] != curr_comp_id:
                            bad_seen_stamp[nidx] = curr_comp_id
                            adj_bad += 1

            max_freedom = max(max_freedom, comp_size + adj_bad)
            curr_comp_id += 1

    print(max_freedom)


if __name__ == "__main__":
    main()
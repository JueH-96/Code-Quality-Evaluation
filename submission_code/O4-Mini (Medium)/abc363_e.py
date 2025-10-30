import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    H = int(line[0]); W = int(line[1]); Y = int(line[2])
    N = H * W
    A = [0] * N
    A_max = 0
    idx = 0
    for _ in range(H):
        row = data.readline().split()
        for v in row:
            ai = int(v)
            A[idx] = ai
            if ai > A_max: A_max = ai
            idx += 1

    # Dijkstra-like to compute flood threshold dist[idx] for each cell
    INF = 10**9
    dist = [INF] * N
    import heapq
    hq = []

    # initialize border cells
    # top row 0..W-1
    for j in range(W):
        d0 = A[j]
        dist[j] = d0
        hq.append((d0, j))
    # bottom row
    base = W*(H-1)
    for j in range(W):
        idx0 = base + j
        d0 = A[idx0]
        if d0 < dist[idx0]:
            dist[idx0] = d0
            hq.append((d0, idx0))
    # left and right columns, excluding corners
    for i in range(1, H-1):
        idx_l = i * W
        d_l = A[idx_l]
        if d_l < dist[idx_l]:
            dist[idx_l] = d_l
            hq.append((d_l, idx_l))
        idx_r = i * W + (W-1)
        d_r = A[idx_r]
        if d_r < dist[idx_r]:
            dist[idx_r] = d_r
            hq.append((d_r, idx_r))

    heapq.heapify(hq)
    # neighbors: up/down/left/right
    while hq:
        d, u = heapq.heappop(hq)
        if d != dist[u]:
            continue
        i = u // W
        j = u - i*W
        nd = 0
        # up
        if i > 0:
            v = u - W
            nd = d if d >= A[v] else A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))
        # down
        if i < H-1:
            v = u + W
            nd = d if d >= A[v] else A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))
        # left
        if j > 0:
            v = u - 1
            nd = d if d >= A[v] else A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))
        # right
        if j < W-1:
            v = u + 1
            nd = d if d >= A[v] else A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

    # count flood thresholds
    # max threshold needed up to max(A_max, Y)
    MX = A_max if A_max > Y else Y
    cnt = [0] * (MX + 2)
    for v in dist:
        # v is flood level: cell floods when sea level >= v
        if v <= MX:
            cnt[v] += 1
        else:
            # if v > MX, it'll never be <= y for y in [1..Y], so safe to ignore
            pass

    # prefix sum of counts
    pref = [0] * (MX + 2)
    running = 0
    for lev in range(MX + 1):
        running += cnt[lev]
        pref[lev] = running

    total = N
    out = []
    # for years y = 1..Y
    for y in range(1, Y+1):
        if y <= MX:
            flooded = pref[y]
        else:
            flooded = pref[MX]
        remain = total - flooded
        out.append(str(remain))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
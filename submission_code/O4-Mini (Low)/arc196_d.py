import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N, M, Q = map(int, input().split())
    intervals = []
    for i in range(M):
        s, t = map(int, input().split())
        l = min(s, t)
        r = max(s, t)
        intervals.append((l, r, i))
    # bad[i] = minimal j index that crosses with i, or INF
    INF = M + 5
    bad = [INF] * M
    # sort by l
    intervals.sort(key=lambda x: (x[0], x[1]))
    import heapq
    # active heap of (r, idx)
    active = []
    # sweep by increasing l
    for l_b, r_b, idx_b in intervals:
        # remove expired: those with r <= l_b
        while active and active[0][0] <= l_b:
            heapq.heappop(active)
        # detect crossings: while minimal r_a < r_b
        # active.r all > l_b
        while active and active[0][0] < r_b:
            r_a, idx_a = heapq.heappop(active)
            # they cross: l_a < l_b < r_a < r_b
            # update bad for input-order pair
            if idx_a < idx_b:
                if bad[idx_a] > idx_b:
                    bad[idx_a] = idx_b
            else:
                if bad[idx_b] > idx_a:
                    bad[idx_b] = idx_a
            # we remove a so not considered further
        # add b to active
        heapq.heappush(active, (r_b, idx_b))
    # build segment tree on bad for range minimum
    size = 1
    while size < M:
        size <<= 1
    seg = [INF] * (2 * size)
    for i in range(M):
        seg[size + i] = bad[i]
    for i in range(size - 1, 0, -1):
        seg[i] = seg[2 * i] if seg[2 * i] < seg[2 * i + 1] else seg[2 * i + 1]
    def range_min(l, r):
        # inclusive l,r
        l += size
        r += size
        m = INF
        while l <= r:
            if (l & 1) == 1:
                if seg[l] < m: m = seg[l]
                l += 1
            if (r & 1) == 0:
                if seg[r] < m: m = seg[r]
                r -= 1
            l //= 2; r //= 2
        return m
    out = []
    for _ in range(Q):
        L, R = map(int, input().split())
        L -= 1; R -= 1
        mn = range_min(L, R)
        # must have no bad_i <= R, so mn > R
        out.append("Yes" if mn > R else "No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
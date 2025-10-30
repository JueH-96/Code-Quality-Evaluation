import sys
import threading
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    # read X and build A[i] = X[i] - i
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        x = int(next(it))
        A[i] = x - i
    # segment tree arrays
    size4 = 4 * N
    seg_max = [0] * size4
    seg_sum = [0] * size4
    seg_tag = [None] * size4
    sys.setrecursionlimit(10**7)
    # build initial tree
    def build(node, l, r):
        if l == r:
            v = A[l]
            seg_max[node] = v
            seg_sum[node] = v
        else:
            mid = (l + r) >> 1
            lc = node << 1
            rc = lc | 1
            build(lc, l, mid)
            build(rc, mid+1, r)
            # combine
            mv_l = seg_max[lc]; mv_r = seg_max[rc]
            seg_max[node] = mv_l if mv_l >= mv_r else mv_r
            seg_sum[node] = seg_sum[lc] + seg_sum[rc]
    build(1, 1, N)
    # push lazy tag to children
    def push(node, l, r):
        v = seg_tag[node]
        if v is not None:
            lc = node << 1
            rc = lc | 1
            mid = (l + r) >> 1
            # left child
            seg_tag[lc] = v
            seg_max[lc] = v
            seg_sum[lc] = v * (mid - l + 1)
            # right child
            seg_tag[rc] = v
            seg_max[rc] = v
            seg_sum[rc] = v * (r - mid)
            # clear
            seg_tag[node] = None
    # range assign [ql..qr] = v
    def range_assign(node, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            seg_tag[node] = v
            seg_max[node] = v
            seg_sum[node] = v * (r - l + 1)
            return
        push(node, l, r)
        mid = (l + r) >> 1
        lc = node << 1
        rc = lc | 1
        if ql <= mid:
            range_assign(lc, l, mid, ql, qr, v)
        if qr > mid:
            range_assign(rc, mid+1, r, ql, qr, v)
        # pull up
        mv_l = seg_max[lc]; mv_r = seg_max[rc]
        seg_max[node] = mv_l if mv_l >= mv_r else mv_r
        seg_sum[node] = seg_sum[lc] + seg_sum[rc]
    # range sum query on [ql..qr]
    def range_sum(node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return seg_sum[node]
        push(node, l, r)
        mid = (l + r) >> 1
        s = 0
        lc = node << 1
        rc = lc | 1
        if ql <= mid:
            s += range_sum(lc, l, mid, ql, qr)
        if qr > mid:
            s += range_sum(rc, mid+1, r, ql, qr)
        return s
    # find first index i in [l..r] with A[i] >= val; if none return N+1
    def find_first_geq(node, l, r, val):
        if seg_max[node] < val:
            return N + 1
        if l == r:
            return l
        push(node, l, r)
        mid = (l + r) >> 1
        lc = node << 1
        # if left child has a >=val, go left
        if seg_max[lc] >= val:
            return find_first_geq(lc, l, mid, val)
        else:
            return find_first_geq(lc|1, mid+1, r, val)
    # process queries
    Q = int(next(it))
    ans = 0
    for _ in range(Q):
        T = int(next(it))
        G = int(next(it))
        C = G - T
        # find first position j0 with A[j0] >= C
        j0 = find_first_geq(1, 1, N, C)
        if j0 > T:
            # move block to the right from T..r
            # r = j0-1 (if j0>N => r=N)
            if j0 <= N:
                r = j0 - 1
            else:
                r = N
            # sum of A[T..r]
            s = range_sum(1, 1, N, T, r)
            # cost = sum(C - A[i])
            cnt = r - T + 1
            ans += C * cnt - s
            # assign A[T..r] = C
            range_assign(1, 1, N, T, r, C)
        else:
            # j0 <= T means A[T] >= C
            # find first pos l0 with A[l0] >= C+1
            l0 = find_first_geq(1, 1, N, C + 1)
            if l0 <= T:
                # move block to the left from l..T
                l = l0
                s = range_sum(1, 1, N, l, T)
                cnt = T - l + 1
                ans += s - C * cnt
                range_assign(1, 1, N, l, T, C)
            # else A[T] == C, no moves
    # output
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()
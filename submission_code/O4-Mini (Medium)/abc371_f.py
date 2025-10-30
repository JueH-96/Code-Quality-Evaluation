import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin
    N = int(data.readline())
    Xs = list(map(int, data.readline().split()))
    Q = int(data.readline())
    # Build initial A: A[i] = Xs[i-1] - i
    N4 = 4 * N + 10
    tree_sum = [0] * N4
    tree_mn = [0] * N4
    tree_mx = [0] * N4
    tag_val = [0] * N4
    tag_has = [False] * N4
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = Xs[i-1] - i

    def apply_tag(node, l, r, v):
        tree_sum[node] = v * (r - l + 1)
        tree_mn[node] = v
        tree_mx[node] = v
        tag_val[node] = v
        tag_has[node] = True

    def push(node, l, r):
        if tag_has[node]:
            mid = (l + r) >> 1
            lc = node << 1
            rc = lc | 1
            v = tag_val[node]
            apply_tag(lc, l, mid, v)
            apply_tag(rc, mid + 1, r, v)
            tag_has[node] = False

    def pull(node):
        lc = node << 1
        rc = lc | 1
        tree_sum[node] = tree_sum[lc] + tree_sum[rc]
        # since A non-decreasing, mn from left, mx from right
        if tree_mn[lc] < tree_mn[rc]:
            tree_mn[node] = tree_mn[lc]
        else:
            tree_mn[node] = tree_mn[lc]  # leftmost is min anyway
        if tree_mx[rc] > tree_mx[lc]:
            tree_mx[node] = tree_mx[rc]
        else:
            tree_mx[node] = tree_mx[rc]

    def build(node, l, r):
        if l == r:
            v = A[l]
            tree_sum[node] = v
            tree_mn[node] = v
            tree_mx[node] = v
        else:
            mid = (l + r) >> 1
            lc = node << 1
            rc = lc | 1
            build(lc, l, mid)
            build(rc, mid + 1, r)
            pull(node)

    def range_assign(node, l, r, ql, qr, v):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            apply_tag(node, l, r, v)
            return
        push(node, l, r)
        mid = (l + r) >> 1
        lc = node << 1
        rc = lc | 1
        if ql <= mid:
            range_assign(lc, l, mid, ql, qr, v)
        if qr > mid:
            range_assign(rc, mid + 1, r, ql, qr, v)
        pull(node)

    def range_sum_q(node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return tree_sum[node]
        push(node, l, r)
        mid = (l + r) >> 1
        s = 0
        if ql <= mid:
            s += range_sum_q(node << 1, l, mid, ql, qr)
        if qr > mid:
            s += range_sum_q((node << 1) | 1, mid + 1, r, ql, qr)
        return s

    INF = N + 1

    def find_first_ge(node, l, r, ql, qr, x):
        # first index in [ql..qr] with A[j] >= x
        if qr < l or r < ql or tree_mx[node] < x:
            return INF
        if l == r:
            return l
        push(node, l, r)
        mid = (l + r) >> 1
        lc = node << 1
        rc = lc | 1
        res = INF
        if ql <= mid:
            res = find_first_ge(lc, l, mid, ql, qr, x)
        if res != INF:
            return res
        if qr > mid:
            res = find_first_ge(rc, mid + 1, r, ql, qr, x)
        return res

    def find_first_gt(node, l, r, ql, qr, x):
        # first index in [ql..qr] with A[j] > x
        if qr < l or r < ql or tree_mn[node] > x and False:
            # this condition is useless; real prune:
            pass
        if qr < l or r < ql or tree_mx[node] <= x:
            return INF
        if l == r:
            return l
        push(node, l, r)
        mid = (l + r) >> 1
        lc = node << 1
        rc = lc | 1
        res = INF
        if ql <= mid:
            res = find_first_gt(lc, l, mid, ql, qr, x)
        if res != INF:
            return res
        if qr > mid:
            res = find_first_gt(rc, mid + 1, r, ql, qr, x)
        return res

    # build initial tree
    build(1, 1, N)

    ans = 0
    for _ in range(Q):
        parts = data.readline().split()
        if not parts:
            parts = data.readline().split()
        t = int(parts[0])
        g = int(parts[1])
        x = g - t
        # get current A[t]
        # cur = range_sum_q(1,1,N,t,t)
        # Or faster: use a point min instead of sum:
        cur = range_sum_q(1, 1, N, t, t)
        if cur >= x:
            # left case
            L = find_first_ge(1, 1, N, 1, t, x)
            # L always <= t
            sum_L = range_sum_q(1, 1, N, L, t)
            cnt = t - L + 1
            ans += sum_L - cnt * x
            range_assign(1, 1, N, L, t, x)
        else:
            # right case
            R = find_first_gt(1, 1, N, t + 1, N, x)
            if R == INF:
                R = N + 1
            # sum over [t..R-1]
            sum_R = range_sum_q(1, 1, N, t, R - 1)
            cnt = R - t
            ans += cnt * x - sum_R
            range_assign(1, 1, N, t, R - 1, x)
    # final answer
    print(ans)

if __name__ == "__main__":
    main()
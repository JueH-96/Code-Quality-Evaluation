import threading
import sys
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    class Solution:
        def countNonDecreasingSubarrays(self, nums, k):
            n = len(nums)
            # Build segment tree for f[0..n-1], initial f[0]=0, rest = INF
            INF = 10**30
            # size = power of two >= n
            size = 1
            while size < n:
                size <<= 1
            N = size
            # tree_min and lazy arrays size 2*N
            tree_min = [INF] * (2 * N)
            lazy = [0] * (2 * N)
            # initialize leaves
            # f[0] = 0
            if n > 0:
                tree_min[N + 0] = 0
            for i in range(1, n):
                tree_min[N + i] = INF
            # build internal nodes
            for i in range(N - 1, 0, -1):
                tree_min[i] = tree_min[2 * i] if tree_min[2 * i] < tree_min[2 * i + 1] else tree_min[2 * i + 1]
            # lazy initialized to 0
            # functions
            def _apply(idx, val):
                tree_min[idx] += val
                lazy[idx] += val
            def _push(idx):
                v = lazy[idx]
                if v:
                    _apply(idx * 2, v)
                    _apply(idx * 2 + 1, v)
                    lazy[idx] = 0
            # range add val on [lq..rq]
            def range_add(lq, rq, val, idx=1, left=0, right=N-1):
                if lq <= left and right <= rq:
                    tree_min[idx] += val
                    lazy[idx] += val
                    return
                if rq < left or right < lq:
                    return
                _push(idx)
                mid = (left + right) >> 1
                lc = idx * 2; rc = lc + 1
                if lq <= mid:
                    range_add(lq, rq, val, lc, left, mid)
                if rq > mid:
                    range_add(lq, rq, val, rc, mid+1, right)
                # update self
                tree_min[idx] = tree_min[lc] if tree_min[lc] < tree_min[rc] else tree_min[rc]
            # range min query [lq..rq]
            def range_min(lq, rq, idx=1, left=0, right=N-1):
                if lq <= left and right <= rq:
                    return tree_min[idx]
                if rq < left or right < lq:
                    return INF
                _push(idx)
                mid = (left + right) >> 1
                res = INF
                if lq <= mid:
                    v = range_min(lq, rq, idx*2, left, mid)
                    if v < res: res = v
                if rq > mid:
                    v = range_min(lq, rq, idx*2+1, mid+1, right)
                    if v < res: res = v
                return res
            # find first i in [0..N-1] s.t. f[i] <= k
            def find_first_leq(k_val, idx=1, left=0, right=N-1):
                if tree_min[idx] > k_val:
                    return None
                # if leaf
                if left == right:
                    return left
                _push(idx)
                mid = (left + right) >> 1
                lc = idx * 2; rc = lc + 1
                if tree_min[lc] <= k_val:
                    return find_first_leq(k_val, lc, left, mid)
                else:
                    return find_first_leq(k_val, rc, mid+1, right)
            res = 0
            # M segments: list of (value, count), covers l in [0..r-1]
            st = []
            # process r
            for r, x in enumerate(nums):
                if r == 0:
                    # new position 0: f[0] already 0
                    # M segments init
                    st = [(x, 1)]
                else:
                    # f update for l in [0..r-1]: for segments with val > x
                    # use old st
                    pos_start = 0
                    for val, cnt in st:
                        if val > x:
                            # update f in [pos_start .. pos_start+cnt-1] by (val - x)
                            delta = val - x
                            # perform range add
                            range_add(pos_start, pos_start + cnt - 1, delta)
                            pos_start += cnt
                        else:
                            break
                    # set f[r] = 0: old value = INF; so we can do point update
                    # get old
                    # old = range_min(r,r)
                    # but old is INF; we want f[r] = 0; so delta = -old => huge, but INF too big
                    # Instead, we know that f[r] has never been updated (never in l<r updates),
                    # so in tree it is INF. We can set it by a direct path update:
                    # do range_add(r,r,-INF) + ??? But INF is 1e30, safe to subtract.
                    # simpler: read old and subtract.
                    old = range_min(r, r)
                    if old != 0:
                        # delta = -old
                        range_add(r, r, -old)
                    # update M segments with new element x
                    sum_cnt = 0
                    # pop small
                    # pop segments with val < x
                    while st and st[-1][0] < x:
                        sum_cnt += st[-1][1]
                        st.pop()
                    # merge if equal
                    if st and st[-1][0] == x:
                        # merge counts
                        prev_cnt = st[-1][1]
                        st[-1] = (x, prev_cnt + sum_cnt + 1)
                    else:
                        st.append((x, sum_cnt + 1))
                # find L(r)
                idx0 = find_first_leq(k)
                if idx0 is None:
                    # no l with f[l] <= k => no valid
                    continue
                # ensure idx0 <= r
                if idx0 > r:
                    # next valid always r
                    idx0 = r
                res += (r - idx0 + 1)
            return res

    # If needed to read input, but here we just define the class.
    # The code above is self-contained for the method.
    # No I/O as per instructions.
    # However, to allow testing, you might un-comment:
    # sol = Solution()
    # print(sol.countNonDecreasingSubarrays([...], k))

    # For online judges that call via input, you can adapt.


    # Example simple test:
    # sol = Solution()
    # print(sol.countNonDecreasingSubarrays([6,3,1,2,4,4], 7))  # expect 17
    # print(sol.countNonDecreasingSubarrays([6,3,1,3,6], 4))    # expect 12

threading.Thread(target=main).start()
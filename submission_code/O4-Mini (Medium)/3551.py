import threading
import sys
def main():
    import sys
    sys.setrecursionlimit(1000000)
    from typing import List

    class SegTree:
        __slots__ = ('n', 'mx1', 'mx2', 'cnt', 'tag')
        def __init__(self, n, INF):
            self.n = n
            size = 4 * n
            self.mx1 = [INF] * size
            self.mx2 = [-INF] * size
            self.cnt = [0] * size
            self.tag = [INF] * size
            self._build(1, 0, n - 1, INF)

        def _build(self, k, l, r, INF):
            # initialize all b[i] = INF
            self.mx1[k] = INF
            self.mx2[k] = -INF
            self.cnt[k] = (r - l + 1)
            self.tag[k] = INF
            if l == r:
                return
            m = (l + r) // 2
            self._build(k*2,   l,   m, INF)
            self._build(k*2+1, m+1, r, INF)

        def _push(self, k):
            tg = self.tag[k]
            if tg == float('inf'):
                return
            for c in (k*2, k*2+1):
                if self.mx1[c] > tg:
                    # apply chmin to child
                    self.mx1[c] = tg
                    if self.tag[c] > tg:
                        self.tag[c] = tg
            self.tag[k] = float('inf')

        def _pull(self, k):
            lch, rch = k*2, k*2+1
            # max1
            if self.mx1[lch] >= self.mx1[rch]:
                self.mx1[k] = self.mx1[lch]
                self.cnt[k] = self.cnt[lch]
            else:
                self.mx1[k] = self.mx1[rch]
                self.cnt[k] = self.cnt[rch]
            if self.mx1[lch] == self.mx1[k]:
                self.cnt[k] += (self.cnt[lch] if self.mx1[lch] == self.mx1[k] else 0)
            if self.mx1[rch] == self.mx1[k]:
                self.cnt[k] += (self.cnt[rch] if self.mx1[rch] == self.mx1[k] else 0)
            # compute second max
            cands = []
            # from left child
            if self.mx1[lch] == self.mx1[k]:
                cands.append(self.mx2[lch])
            else:
                cands.append(self.mx1[lch])
            # from right child
            if self.mx1[rch] == self.mx1[k]:
                cands.append(self.mx2[rch])
            else:
                cands.append(self.mx1[rch])
            self.mx2[k] = cands[0] if cands[0] >= cands[1] else cands[1]

        def range_chmin(self, ql, qr, x):
            self._range_chmin(1, 0, self.n - 1, ql, qr, x)

        def _range_chmin(self, k, l, r, ql, qr, x):
            if self.mx1[k] <= x or qr < l or r < ql:
                return
            if ql <= l and r <= qr and self.mx2[k] < x:
                # we can apply tag here
                self.mx1[k] = x
                if self.tag[k] > x:
                    self.tag[k] = x
                return
            self._push(k)
            m = (l + r) // 2
            self._range_chmin(k*2,   l,   m, ql, qr, x)
            self._range_chmin(k*2+1, m+1, r, ql, qr, x)
            self._pull(k)

        def point_query(self, pos):
            return self._point_query(1, 0, self.n - 1, pos)

        def _point_query(self, k, l, r, pos):
            if l == r:
                return self.mx1[k]
            self._push(k)
            m = (l + r) // 2
            if pos <= m:
                return self._point_query(k*2,   l,   m, pos)
            else:
                return self._point_query(k*2+1, m+1, r, pos)

    class Solution:
        def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
            n = len(nums)
            # 1) compute f[s][e] = XOR-score of nums[s..e]
            f = [ [0]*n for _ in range(n) ]
            for i in range(n):
                f[i][i] = nums[i]
            for length in range(2, n+1):
                # compute all s,e with e = s+length-1
                for s in range(0, n-length+1):
                    e = s + length - 1
                    # recurrence f[s][e] = f[s][e-1] ^ f[s+1][e]
                    f[s][e] = f[s][e-1] ^ f[s+1][e]

            # 2) group queries by right endpoint
            qs = [[] for _ in range(n)]
            for idx, (L, R) in enumerate(queries):
                qs[R].append((L, idx))

            # 3) segment tree on b[L] = -best[L], with initial b[L]=+INF
            INF = 1 << 62
            st = SegTree(n, INF)
            ans = [0] * len(queries)

            # 4) sweep R from 0..n-1
            for R in range(n):
                # for each start s<=R, insert subarray [s..R] with value v=f[s][R]
                for s in range(R+1):
                    v = f[s][R]
                    b = -v
                    # do b[L] = min(b[L], b) for L in [0..s]
                    st.range_chmin(0, s, b)

                # answer queries ending at R
                for (L, qi) in qs[R]:
                    bval = st.point_query(L)
                    ans[qi] = -bval

            return ans

    # If reading input from stdin is required, put parsing here.
    # Otherwise, this module defines Solution.maximumSubarrayXor() as requested.

    # Example usage (uncomment to test):
    # sol = Solution()
    # print(sol.maximumSubarrayXor([2,8,4,32,16,1], [[0,2],[1,4],[0,5]]))  # [12,60,60]

    # For code-run platforms, we could read input and print output:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    nums = [int(next(it)) for _ in range(n)]
    q = int(next(it))
    queries = []
    for _ in range(q):
        l = int(next(it)); r = int(next(it))
        queries.append((l, r))
    sol = Solution()
    res = sol.maximumSubarrayXor(nums, queries)
    print('
'.join(map(str, res)))

if __name__ == "__main__":
    threading.Thread(target=main).start()
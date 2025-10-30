def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    Q, sh, sw = map(int, input().split())
    sh -= 1
    sw -= 1

    # We'll flatten the grid in order of increasing (h+w).
    # For each cell (h,w), define idx[h][w] = position in that ordering.
    idx = [[0]*W for _ in range(H)]
    order = []
    for s in range(H+W-1):
        for h in range(H):
            w = s - h
            if 0 <= w < W:
                idx[h][w] = len(order)
                order.append((h, w))
    N = len(order)

    #
    # We build a segment tree that stores the DP value for each cell in this
    # linear order.  We will support:
    #  - range-multiply by a factor M on a suffix [p, N)
    #  - point-update at a single position
    #  - point-query
    #
    # The DP recurrence is:
    #   dp[h,w] = A[h][w] * (dp[h-1,w] + dp[h,w-1]),
    # where out-of-bounds dp is treated as 0 except dp[0,0]=A[0][0].
    #
    # When we change A[x][y], we must re-multiply every dp[h,w] for h>=x,w>=y
    # by (newA/oldA), because each of those DP values is linear in A[x][y].
    # Then we must also re-compute dp[x][y] itself as
    #   A[x][y] * (dp[x-1,y] + dp[x,y-1]),
    # and store that in its position.  That's a point-update.
    #
    # Finally, the answer S is dp[H-1,W-1] at the last index.

    class SegTree:
        def __init__(self, n):
            self.n = 1
            while self.n < n:
                self.n <<= 1
            # store value, and lazy-mul
            self.val = [0] * (2*self.n)
            self.laz = [1] * (2*self.n)

        def apply_mul(self, i, mul):
            self.val[i] = self.val[i] * mul % MOD
            self.laz[i] = self.laz[i] * mul % MOD

        def push(self, i):
            if self.laz[i] != 1:
                self.apply_mul(i*2,   self.laz[i])
                self.apply_mul(i*2+1, self.laz[i])
                self.laz[i] = 1

        def pull(self, i):
            self.val[i] = (self.val[2*i] + self.val[2*i+1]) % MOD

        # range-multiply on [l, r)
        def range_mul(self, l, r, mul, i=1, lo=0, hi=None):
            if hi is None:
                hi = self.n
            if r <= lo or hi <= l:
                return
            if l <= lo and hi <= r:
                self.apply_mul(i, mul)
                return
            self.push(i)
            mid = (lo + hi)//2
            self.range_mul(l, r, mul, 2*i,   lo, mid)
            self.range_mul(l, r, mul, 2*i+1, mid, hi)
            self.pull(i)

        # point-assign: set pos -> x
        def point_set(self, pos, x):
            i = 1; lo = 0; hi = self.n
            # descend
            while hi - lo > 1:
                self.push(i)
                mid = (lo + hi)//2
                if pos < mid:
                    i = 2*i; hi = mid
                else:
                    i = 2*i+1; lo = mid
            self.val[i] = x
            self.laz[i] = 1
            # ascend
            i >>= 1
            while i:
                self.pull(i)
                i >>= 1

        # point-query at pos
        def point_get(self, pos):
            i = 1; lo = 0; hi = self.n
            while hi - lo > 1:
                self.push(i)
                mid = (lo + hi)//2
                if pos < mid:
                    i = 2*i; hi = mid
                else:
                    i = 2*i+1; lo = mid
            return self.val[i]

    st = SegTree(N)

    # Initialize the DP array in linear order.
    # We compute dp in the anti-diagonal order:
    dp = [0]*N
    for k, (h,w) in enumerate(order):
        if h==0 and w==0:
            dp[k] = A[0][0]
        else:
            s = 0
            if h>0:
                s = (s + dp[idx[h-1][w]])%MOD
            if w>0:
                s = (s + dp[idx[h][w-1]])%MOD
            dp[k] = A[h][w] * s % MOD
        st.point_set(k, dp[k])

    # Process queries
    cur_h, cur_w = sh, sw
    for _ in range(Q):
        d,a = input().split()
        a = int(a)
        # move
        if d=='L': cur_w -=1
        elif d=='R': cur_w +=1
        elif d=='U': cur_h -=1
        else:          cur_h +=1
        # we will update A[cur_h][cur_w] from old->a
        old = A[cur_h][cur_w]
        A[cur_h][cur_w] = a
        pos = idx[cur_h][cur_w]

        if old != 0:
            inv_old = pow(old, MOD-2, MOD)
            factor = a * inv_old % MOD
        else:
            # old=0 -> everything that used it was 0; we must rebuild that cell's dp
            # into a*(dp[h-1,w]+dp[h,w-1]).  For the suffix multiplication we skip if old=0.
            factor = 1

        # 1) multiply dp[k] by factor for all k in [pos, N)
        if factor != 1:
            st.range_mul(pos, N, factor)

        # 2) re-compute dp at (cur_h,cur_w) itself
        #    dp = A * (dp from top + dp from left)
        left_val  = st.point_get(idx[cur_h][cur_w-1]) if cur_w>0 else 0
        up_val    = st.point_get(idx[cur_h-1][cur_w]) if cur_h>0 else 0
        new_dp = a * (left_val + up_val) % MOD

        # 3) point-set it
        st.point_set(pos, new_dp)

        # Finally, query dp[H-1,W-1]
        ans = st.point_get(idx[H-1][W-1])
        print(ans)


if __name__ == "__main__":
    main()
from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        # Li Chao tree for maximum of lines: y = m*x + b, x in [0, n-1].
        n = len(nums)
        if n == 1:
            return 0
        
        # Size of segment tree: 4*n is enough.
        size = 4 * n
        # st_m[i], st_b[i] define the line at node i: y = st_m[i] * x + st_b[i].
        st_m = [0] * size
        # initialize b to a very large negative so empty lines give -inf
        NEG_INF = -10**30
        st_b = [NEG_INF] * size
        
        # Add a new line (m_new, b_new) into the Li Chao tree
        def add_line(m_new: int, b_new: int, node: int = 1, l: int = 0, r: int = None):
            if r is None:
                r = n - 1
            # Current node's line: f(x) = m*x + b
            m_cur, b_cur = st_m[node], st_b[node]
            mid = (l + r) // 2
            
            # At x=mid, if new line is better, swap it with the current
            if m_new * mid + b_new > m_cur * mid + b_cur:
                st_m[node], m_new = m_new, m_cur
                st_b[node], b_new = b_new, b_cur
                # now (m_new, b_new) is the worse line at mid
            
            # If leaf, we're done
            if l == r:
                return
            
            # Decide which side the worse line might beat the stored line
            if m_new * l + b_new > st_m[node] * l + st_b[node]:
                # new line (worse at mid but better at l) goes left
                add_line(m_new, b_new, node * 2, l, mid)
            elif m_new * r + b_new > st_m[node] * r + st_b[node]:
                # better at r, goes right
                add_line(m_new, b_new, node * 2 + 1, mid + 1, r)
            # else: it's worse everywhere in [l,r], discard
        
        # Query the max value at x = qx
        def query(qx: int, node: int = 1, l: int = 0, r: int = None) -> int:
            if r is None:
                r = n - 1
            # value of this node's line
            res = st_m[node] * qx + st_b[node]
            if l == r:
                return res
            mid = (l + r) // 2
            # go to the side containing qx
            if qx <= mid:
                return max(res, query(qx, node * 2, l, mid))
            else:
                return max(res, query(qx, node * 2 + 1, mid + 1, r))
        
        # dp[0] = 0; initial line from i=0: m = nums[0], b = dp[0] - 0*nums[0] = 0
        add_line(nums[0], 0)
        dp_last = 0
        
        # Compute dp[j] for j = 1..n-1
        for j in range(1, n):
            # dp[j] = max over i<j of dp[i] + (j-i)*nums[i]
            #      = max_i ( nums[i]*j + (dp[i] - i*nums[i]) )
            best = query(j)
            # store dp[j]
            dp_last = best
            # line for this j: m = nums[j], b = dp[j] - j*nums[j]
            add_line(nums[j], best - j * nums[j])
        
        return dp_last
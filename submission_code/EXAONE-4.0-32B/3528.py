import sys
sys.setrecursionlimit(300000)

class Line:
    __slots__ = ('a', 'b')
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, x):
        return self.a * x + self.b

class LiChaoTree:
    __slots__ = ('L', 'R', 'tree', 'size')
    def __init__(self, L, R):
        self.L = L
        self.R = R
        self.size = 4 * (R - L + 1)
        self.tree = [None] * self.size
        
    def update(self, a, b):
        line = Line(a, b)
        self._update(1, self.L, self.R, line)
        
    def _update(self, node, l, r, line):
        if l > r:
            return
        if l == r:
            if self.tree[node] is None or line(l) > self.tree[node](l):
                self.tree[node] = line
            return
        
        m = (l + r) // 2
        curr_line = self.tree[node]
        if curr_line is None:
            self.tree[node] = line
            return
            
        fl = line(l)
        fc = curr_line(l)
        fm = line(m)
        fcm = curr_line(m)
        
        lef = fl > fc
        mid = fm > fcm
        
        if mid:
            self.tree[node], line = line, curr_line
            
        if lef != mid:
            self._update(2*node, l, m, line)
        else:
            self._update(2*node+1, m+1, r, line)
            
    def query(self, x):
        return self._query(1, self.L, self.R, x)
    
    def _query(self, node, l, r, x):
        if l > r:
            return -10**18
        cur = -10**18
        if self.tree[node] is not None:
            cur = self.tree[node](x)
        if l == r:
            return cur
        m = (l + r) // 2
        if x <= m:
            res = self._query(2*node, l, m, x)
        else:
            res = self._query(2*node+1, m+1, r, x)
        return max(cur, res)

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dp = [0] * n
        dp[n-1] = 0
        L, R = 1, 100000
        tree = LiChaoTree(L, R)
        tree.update(n-1, 0)
        
        for i in range(n-2, -1, -1):
            x = nums[i]
            best = tree.query(x)
            dp[i] = best - i * nums[i]
            tree.update(i, dp[i])
            
        return dp[0]
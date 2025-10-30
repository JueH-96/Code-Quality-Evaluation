class LiChaoTree:
    def __init__(self, xs):
        self.xs = xs
        self.n = len(xs)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [None] * (2 * self.size)
    
    def f(self, line, x):
        a, b = line
        return a * x + b

    def update(self, a, b):
        line = (a, b)
        i = 1
        l = 0
        r = self.size
        while r - l > 0:
            if self.tree[i] is None:
                self.tree[i] = line
                return
            mid = (l + r) // 2
            x_mid = self.xs[mid] if mid < self.n else self.xs[-1]
            if self.f(line, x_mid) > self.f(self.tree[i], x_mid):
                self.tree[i], line = line, self.tree[i]
            xl = self.xs[l]
            xr = self.xs[r-1] if r-1 < self.n else self.xs[-1]
            if self.f(line, xl) > self.f(self.tree[i], xl):
                i = 2 * i
                r = mid
            else:
                i = 2 * i + 1
                l = mid

    def query(self, x):
        i = 1
        l = 0
        r = self.size
        res = -10**18
        while i < 2 * self.size:
            if self.tree[i] is not None:
                res = max(res, self.f(self.tree[i], x))
            if r - l == 1:
                break
            mid = (l + r) // 2
            if x < self.xs[mid]:
                i = 2 * i
                r = mid
            else:
                i = 2 * i + 1
                l = mid
        return res

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        max_val = max(nums)
        xs = list(range(1, max_val + 1))
        tree = LiChaoTree(xs)
        dp = [0] * n
        dp[n-1] = 0
        tree.update(n-1, dp[n-1])
        for i in range(n-2, -1, -1):
            x = nums[i]
            best = tree.query(x)
            dp[i] = -i * x + best
            tree.update(i, dp[i])
        return dp[0]
from typing import List

# We'll use a Li-Chao Tree to efficiently support queries of the form:
#   dp[j] = max ( for i < j ) { dp[i] + (j - i) * nums[i] }
# which can be rewritten as:
#   dp[j] = max ( for i < j ) { (nums[i]) * j + (dp[i] - i * nums[i]) }
# For each index i, we view "line" f_i(x) = nums[i] * x + (dp[i] - i*nums[i])
# and we want to query the maximum f_i(j) for j. The Li-Chao tree lets us insert lines
# and query maximum value at a given x in O(logN) per query.

class LiChaoTree:
    def __init__(self, l: int, r: int):
        # The tree covers the domain [l, r), with r exclusive.
        self.l = l
        self.r = r
        self.m = None  # slope of the current line
        self.b = None  # intercept of the current line
        self.left = None
        self.right = None

    def eval_line(self, m: int, b: int, x: int) -> int:
        return m * x + b

    def evaluate(self, x: int) -> int:
        # If no line is stored here yet, return a very small number.
        if self.m is None:
            return float('-inf')
        return self.m * x + self.b

    def add_line(self, m: int, b: int):
        self._add_line(m, b, self.l, self.r)

    def _add_line(self, m: int, b: int, l: int, r: int):
        mid = (l + r) // 2
        if self.m is None:
            self.m = m
            self.b = b
            return

        # At the midpoint, decide which line is better.
        if self.eval_line(m, b, mid) > self.eval_line(self.m, self.b, mid):
            # Swap the current line with the new one.
            self.m, m = m, self.m
            self.b, b = b, self.b

        # If the interval is a single point, nothing more to do.
        if r - l == 1:
            return

        # Decide which segment of the interval the new line might be better.
        if self.eval_line(m, b, l) > self.eval_line(self.m, self.b, l):
            if self.left is None:
                self.left = LiChaoTree(l, mid)
            self.left._add_line(m, b, l, mid)
        elif self.eval_line(m, b, r - 1) > self.eval_line(self.m, self.b, r - 1):
            if self.right is None:
                self.right = LiChaoTree(mid, r)
            self.right._add_line(m, b, mid, r)

    def query(self, x: int) -> int:
        return self._query(x, self.l, self.r)

    def _query(self, x: int, l: int, r: int) -> int:
        mid = (l + r) // 2
        res = self.evaluate(x)
        if r - l == 1:
            return res
        if x < mid and self.left:
            res = max(res, self.left._query(x, l, mid))
        elif x >= mid and self.right:
            res = max(res, self.right._query(x, mid, r))
        return res

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0  # Already at the last index, so score is 0.
        
        # dp[i] will store the maximum score achievable to reach index i.
        dp = [0] * n
        dp[0] = 0
        
        # Our x-coordinate domain is the indices we query.
        # We use [0, n) so that every j from 0 to n-1 is covered.
        tree = LiChaoTree(0, n)
        
        # Insert the line corresponding to i = 0.
        # The line is: f_0(x) = nums[0] * x + (dp[0] - 0 * nums[0])
        tree.add_line(nums[0], dp[0])
        
        # Process indices from 1 to n-1.
        for j in range(1, n):
            # Query the maximum value at x = j,
            # which gives dp[j] = max_{i<j} { dp[i] + (j - i) * nums[i] }.
            dp[j] = tree.query(j)
            # Insert the line corresponding to index j for future queries.
            # f_j(x) = nums[j] * x + (dp[j] - j * nums[j])
            tree.add_line(nums[j], dp[j] - j * nums[j])
        
        # The answer is the maximum score to reach the last index.
        return dp[n - 1]
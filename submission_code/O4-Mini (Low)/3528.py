from typing import List

class LiChaoTree:
    class Line:
        __slots__ = ('m', 'b')
        def __init__(self, m: int = 0, b: int = float('-inf')):
            self.m = m
            self.b = b
        def eval(self, x: int) -> int:
            return self.m * x + self.b

    class Node:
        __slots__ = ('line', 'left', 'right')
        def __init__(self, line: 'LiChaoTree.Line'):
            self.line = line
            self.left = None
            self.right = None

    def __init__(self, x_min: int, x_max: int):
        self.x_min = x_min
        self.x_max = x_max
        # Start with a line that returns -inf everywhere
        self.root = LiChaoTree.Node(LiChaoTree.Line(0, float('-inf')))

    def add_line(self, m: int, b: int):
        """
        Add the line y = m*x + b to the Li Chao tree
        """
        new_line = LiChaoTree.Line(m, b)
        self._add_line(self.root, self.x_min, self.x_max, new_line)

    def _add_line(self, node: 'LiChaoTree.Node', l: int, r: int, new_line: 'LiChaoTree.Line'):
        mid = (l + r) // 2
        # At mid-point, ensure node.line is the better one; if not, swap
        if new_line.eval(mid) > node.line.eval(mid):
            node.line, new_line = new_line, node.line
        # If range is a single point, we're done
        if l == r:
            return
        # Now decide which side we might improve
        if new_line.eval(l) > node.line.eval(l):
            # New line is better on the left segment
            if not node.left:
                node.left = LiChaoTree.Node(new_line)
            else:
                self._add_line(node.left, l, mid, new_line)
        elif new_line.eval(r) > node.line.eval(r):
            # New line is better on the right segment
            if not node.right:
                node.right = LiChaoTree.Node(new_line)
            else:
                self._add_line(node.right, mid + 1, r, new_line)

    def query(self, x: int) -> int:
        """
        Query the maximum value at x
        """
        return self._query(self.root, self.x_min, self.x_max, x)

    def _query(self, node: 'LiChaoTree.Node', l: int, r: int, x: int) -> int:
        if not node:
            return float('-inf')
        res = node.line.eval(x)
        if l == r:
            return res
        mid = (l + r) // 2
        if x <= mid:
            return max(res, self._query(node.left, l, mid, x))
        else:
            return max(res, self._query(node.right, mid + 1, r, x))


class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[j] = max over i<j of dp[i] + (j - i) * nums[i]
        # Rewrite: dp[i] - i*nums[i] + j*nums[i], so lines of slope m=nums[i], intercept b=dp[i] - i*nums[i]
        # We query at x=j for maximum value.
        
        # Build a Li Chao tree over x in [0, n-1]
        tree = LiChaoTree(0, n - 1)
        
        # Base case: at i=0, dp[0] = 0, so line is m = nums[0], b = 0 - 0*nums[0] = 0
        tree.add_line(nums[0], 0)
        dp_last = 0
        
        for j in range(1, n):
            # Query the best score to jump into j
            best = tree.query(j)
            dp_last = best
            # Insert the line corresponding to j
            # slope = nums[j], intercept = dp[j] - j*nums[j]
            tree.add_line(nums[j], dp_last - j * nums[j])
        
        return dp_last
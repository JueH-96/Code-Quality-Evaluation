from typing import List

# We'll implement a LiChao Tree to maintain a set of lines of the form f(x) = m*x + b,
# so that we can quickly query the maximum value at a given x.
# In our problem, for each index i we want to consider a line:
#   f_i(x) = nums[i] * x + (dp[i] - i * nums[i])
# Then the best jump to index j is:
#   dp[j] = max_{i<j} f_i(j)
# And when we reach the last index, dp[n-1] is our answer.
#
# The LiChao tree supports building a tree over a range [L, R] for x-values.
# We'll use an iterative LiChao Tree implementation.

class LiChaoTree:
    # Each node stores one line (m, b). We want max over lines.
    class Line:
        __slots__ = 'm', 'b'
        def __init__(self, m: int, b: int):
            self.m = m
            self.b = b
        def value(self, x: int) -> int:
            return self.m * x + self.b

    class Node:
        __slots__ = 'l', 'r', 'line'
        def __init__(self, line: 'LiChaoTree.Line'):
            self.l = None
            self.r = None
            self.line = line

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        # INF line where value always very small
        self.root = None

    def add_line(self, m: int, b: int):
        new_line = LiChaoTree.Line(m, b)
        self.root = self._add_line(self.root, self.lo, self.hi, new_line)
    
    def _add_line(self, node, l, r, new_line: 'LiChaoTree.Line'):
        if node is None:
            return LiChaoTree.Node(new_line)
        mid = (l + r) >> 1
        # Check which line gives higher value at mid.
        if new_line.value(mid) > node.line.value(mid):
            # swap new_line and node.line
            node.line, new_line = new_line, node.line
        if l == r:
            return node
        if new_line.value(l) > node.line.value(l):
            node.l = self._add_line(node.l, l, mid, new_line)
        elif new_line.value(r) > node.line.value(r):
            node.r = self._add_line(node.r, mid+1, r, new_line)
        return node

    def query(self, x: int) -> int:
        return self._query(self.root, self.lo, self.hi, x)
    
    def _query(self, node, l, r, x: int) -> int:
        if node is None:
            return -10**18  # a very small number, lower than any possible score.
        mid = (l + r) >> 1
        res = node.line.value(x)
        if x <= mid:
            res = max(res, self._query(node.l, l, mid, x))
        else:
            res = max(res, self._query(node.r, mid+1, r, x))
        return res
        
class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        # dp[0] = 0, and we start from index 0, no jump needed.
        # We will add the line corresponding to index 0: 
        # f_0(x) = nums[0]*x + (dp[0] - 0 * nums[0]) = nums[0]*x.
        # The domain for x will be indices from 0 to n-1.
        tree = LiChaoTree(0, n-1)
        tree.add_line(nums[0], dp[0] - 0 * nums[0])
        
        for j in range(1, n):
            # For current j, optimal dp is given by query j:
            dp[j] = tree.query(j)
            # Once computed dp[j], add corresponding line for future indices:
            # f_j(x) = nums[j]*x + (dp[j] - j*nums[j])
            tree.add_line(nums[j], dp[j] - j * nums[j])
        return dp[n-1]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.findMaximumScore([1,3,1,5]))  # Output: 7
    print(sol.findMaximumScore([4,3,1,3,2]))  # Output: 16
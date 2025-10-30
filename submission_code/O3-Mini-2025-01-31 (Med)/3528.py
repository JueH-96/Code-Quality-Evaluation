from typing import List

# We'll implement a LiChao Tree to maintain a set of lines of the form f(x) = m * x + b,
# so that we can answer queries for the maximum value at a given x in O(log(range)).
# This will allow us to compute for each index i:
#   dp[i] = -nums[i]*i + max{ dp[j] + nums[i]*j } for j > i
# We'll process from right to left and add lines f(x) = j*x + dp[j].
#
# LiChaoTree implementation:
class LiChaoTree:
    class Line:
        __slots__ = ('m', 'b')
        def __init__(self, m, b):
            self.m = m
            self.b = b
        def get(self, x):
            return self.m * x + self.b

    class Node:
        __slots__ = ('l', 'r', 'line')
        def __init__(self, line):
            self.line = line
            self.l = None
            self.r = None

    def __init__(self, left, right):
        self.left_bound = left
        self.right_bound = right
        # start with a dummy line; here we can use a line that returns -infinity.
        # However, since dp values can be negative, we use a very small number.
        inf_neg = -10**18
        self.root = self.Node(self.Line(0, inf_neg))

    def add_line(self, m, b):
        new_line = self.Line(m, b)
        self._add_line(self.root, self.left_bound, self.right_bound, new_line)

    def _add_line(self, node, l, r, new_line):
        mid = (l + r) // 2
        
        # Check at mid: if new_line gives a better value than current line, swap them.
        if new_line.get(mid) > node.line.get(mid):
            node.line, new_line = new_line, node.line
        
        # If the segment is a single point, nothing else to do.
        if r - l == 1:
            return
        
        if new_line.get(l) > node.line.get(l):
            # the new line is better on the left side
            if node.l is None:
                node.l = self.Node(new_line)
            else:
                self._add_line(node.l, l, mid, new_line)
        elif new_line.get(r - 1) > node.line.get(r - 1):
            # the new line is better on the right side
            if node.r is None:
                node.r = self.Node(new_line)
            else:
                self._add_line(node.r, mid, r, new_line)
        # Else, no segment where new_line is better.

    def query(self, x):
        return self._query(self.root, self.left_bound, self.right_bound, x)

    def _query(self, node, l, r, x):
        if node is None:
            return -10**18  # very small number
        mid = (l + r) // 2
        res = node.line.get(x)
        if r - l == 1:
            return res
        if x < mid and node.l is not None:
            res = max(res, self._query(node.l, l, mid, x))
        elif x >= mid and node.r is not None:
            res = max(res, self._query(node.r, mid, r, x))
        return res

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        # If there's only one index, no jumps are needed.
        if n == 1:
            return 0
        
        # dp[i] is maximum total score if starting at index i (ending at n-1).
        dp = [0] * n
        dp[n-1] = 0  # reaching end from end gives 0 score.
        
        # the query x comes from nums[i]. Given constraints, nums[i] is up to 10^5.
        # We'll build the LiChao tree over x from 0 to 100001 (upper bound exclusive).
        max_x = 100001
        tree = LiChaoTree(0, max_x)
        # Add a line corresponding to j = n-1.
        # For index j, the line is f(x) = j * x + dp[j].
        tree.add_line(n-1, dp[n-1])
        
        # Process indices in reverse order.
        for i in range(n-2, -1, -1):
            # For index i, the recurrence:
            # dp[i] = max_{j > i}[ (j - i) * nums[i] + dp[j] ]
            #       = -nums[i]*i + max_{j > i}[ dp[j] + nums[i]*j ]
            best = tree.query(nums[i])
            dp[i] = best - nums[i] * i
            # Add the line corresponding to the current index i for future queries.
            tree.add_line(i, dp[i])
        return dp[0]
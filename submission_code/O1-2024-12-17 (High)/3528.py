class Solution:
    from typing import List

    def findMaximumScore(self, nums: List[int]) -> int:
        # We'll use a Li Chao Tree to maintain lines of the form:
        #   y = m*x + b
        # so that dp[i] = max( dp[k] + (i - k)*nums[k] ) for k < i
        # can be rewritten as:
        #   dp[i] = max( (dp[k] - k*nums[k]) + i*nums[k] )
        # i.e., dp[i] = max( m * x + b ), where m = nums[k], b = dp[k] - k*nums[k], and x = i.

        class LiChaoTree:
            def __init__(self, left, right):
                self.left_bound = left
                self.right_bound = right
                self.m = None
                self.b = None
                self.left_child = None
                self.right_child = None

            def f(self, x):
                if self.m is None:
                    return float("-inf")
                return self.m * x + self.b

            def _add_line(self, m, b, left, right):
                # If the current node has no line yet, store (m, b) here.
                if self.m is None:
                    self.m = m
                    self.b = b
                    return

                mid = (left + right) // 2

                # If the new line is better at mid, swap them
                if m * mid + b > self.m * mid + self.b:
                    self.m, m = m, self.m
                    self.b, b = b, self.b

                # If we're not at a leaf, figure out which side might need the other line
                if left == right:
                    return
                if m * left + b > self.m * left + self.b:
                    # Go left
                    if self.left_child is None:
                        self.left_child = LiChaoTree(left, mid)
                    self.left_child._add_line(m, b, left, mid)
                else:
                    # Go right
                    if self.right_child is None:
                        self.right_child = LiChaoTree(mid + 1, right)
                    self.right_child._add_line(m, b, mid + 1, right)

            def add_line(self, m, b):
                # Public method to add a line in the range [left_bound, right_bound].
                self._add_line(m, b, self.left_bound, self.right_bound)

            def _query(self, x, left, right):
                # Evaluate the best line stored in this node for x,
                # then descend to left/right child if possible.
                res = self.f(x)
                if left == right or self.m is None:
                    return res
                mid = (left + right) // 2
                if x <= mid and self.left_child is not None:
                    return max(res, self.left_child._query(x, left, mid))
                elif x > mid and self.right_child is not None:
                    return max(res, self.right_child._query(x, mid + 1, right))
                return res

            def query(self, x):
                return self._query(x, self.left_bound, self.right_bound)

        n = len(nums)
        if n == 1:
            # If there's only one index, no jumps are needed, so score is 0.
            return 0

        # Initialize Li Chao Tree over the index range [0..n-1].
        tree = LiChaoTree(0, n - 1)

        # dp[0] = 0; add line based on index 0.
        # We'll keep track of "current_dp" as we go.
        dp0 = 0
        tree.add_line(nums[0], dp0 - 0 * nums[0])

        curr_dp = dp0
        for i in range(1, n):
            # Query the tree for the best possible dp[i].
            curr_dp = tree.query(i)
            # Add a line for future indices based on dp[i] and nums[i].
            tree.add_line(nums[i], curr_dp - i * nums[i])

        return curr_dp
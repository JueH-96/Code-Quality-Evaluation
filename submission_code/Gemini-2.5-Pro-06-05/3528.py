from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0

        # Li Chao Tree for maintaining the upper envelope of a set of lines
        # and querying for the maximum value at a given point.
        # It solves the recurrence:
        # dp[i] = max_{0 <= j < i} { dp[j] + (i - j) * nums[j] }
        # which can be rewritten as:
        # dp[i] = max_{0 <= j < i} { nums[j] * i + (dp[j] - j * nums[j]) }
        # This is equivalent to finding the maximum value at x=i among a set of lines
        # L_j(x) = m_j * x + c_j, where m_j = nums[j] and c_j = dp[j] - j * nums[j].
        class LiChaoTree:
            def __init__(self, n_size):
                self.x_min = 0
                self.x_max = n_size - 1
                # Using a dictionary for a sparse tree representation.
                # Key: node_idx, Value: line (m, c)
                self.tree = {}

            def _eval(self, line, x):
                # A null line evaluates to -infinity.
                if line is None:
                    return -float('inf')
                m, c = line
                return m * x + c

            def add_line(self, line):
                self._add_line_recursive(line, 1, self.x_min, self.x_max)

            def _add_line_recursive(self, line, node_idx, l, r):
                node_line = self.tree.get(node_idx)
                
                # If there's no line at this node, place the new line here.
                if node_line is None:
                    self.tree[node_idx] = line
                    return

                mid = l + (r - l) // 2
                
                # Compare the new line and the node's current line at the midpoint.
                is_line_better_mid = self._eval(line, mid) > self._eval(node_line, mid)
                
                # If the new line is better at the midpoint, it becomes the line for this node.
                # The old line is then pushed down to one of the children.
                if is_line_better_mid:
                    self.tree[node_idx], line = line, node_line
                
                # If we are at a leaf, we can't push down further.
                if l == r:
                    return

                # After a potential swap, self.tree[node_idx] holds the line that's better at mid.
                # `line` holds the one that was worse at mid.
                # We check if `line` can be better on one of the endpoints of the range [l, r].
                # If so, it means there's an intersection, and it could be optimal in a sub-range.
                if self._eval(line, l) > self._eval(self.tree[node_idx], l):
                    # `line` is better at `l` but worse at `mid`. Intersection is in [l, mid).
                    # So, `line` might be optimal in the left child's range.
                    self._add_line_recursive(line, 2 * node_idx, l, mid)
                elif self._eval(line, r) > self._eval(self.tree[node_idx], r):
                    # `line` is worse at `l` and `mid` but better at `r`. Intersection is in (mid, r].
                    # So, `line` might be optimal in the right child's range.
                    self._add_line_recursive(line, 2 * node_idx + 1, mid + 1, r)

            def query(self, x):
                return self._query_recursive(x, 1, self.x_min, self.x_max)

            def _query_recursive(self, x, node_idx, l, r):
                node_line = self.tree.get(node_idx)
                # The best score at 'x' is the max of the line at the current node
                # and the best line from the relevant child's subtree.
                res = self._eval(node_line, x)
                
                if l == r:
                    return res

                mid = l + (r - l) // 2
                if x <= mid:
                    res = max(res, self._query_recursive(x, 2 * node_idx, l, mid))
                else:
                    res = max(res, self._query_recursive(x, 2 * node_idx + 1, mid + 1, r))
                
                return res

        dp = [0] * n
        lct = LiChaoTree(n)

        # Base case: At index 0, score is 0. This defines the first line L_0.
        # m_0 = nums[0], c_0 = dp[0] - 0 * nums[0] = 0.
        m0 = nums[0]
        c0 = 0
        lct.add_line((m0, c0))
        
        for i in range(1, n):
            # dp[i] is the max score to reach index i. This is found by querying
            # the upper envelope of lines L_0...L_{i-1} at x=i.
            dp[i] = lct.query(i)
            
            # Create a new line L_i for index i and add it to the LCT.
            # m_i = nums[i], c_i = dp[i] - i * nums[i]
            m_i = nums[i]
            c_i = dp[i] - i * nums[i]
            lct.add_line((m_i, c_i))
            
        return dp[n - 1]
class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        """
        We want to compute a DP where:
            dp[i] = maximum score achievable upon reaching index i.
        
        A jump from index i to index j (j > i) contributes (j - i)*nums[i] to the score.
        Thus:
            dp[j] = max over i < j of [ dp[i] + (j - i)*nums[i] ]
                   = max over i < j of [ dp[i] + (j*nums[i] - i*nums[i]) ]
                   = max over i < j of [ (dp[i] - i*nums[i]) + j*nums[i] ].
        
        If we define:
            line_i(x) = nums[i]*x + (dp[i] - i*nums[i]),
        then dp[j] is the maximum value of line_i(j) over all i < j.
        
        We can maintain a data structure (a "Li Chao Tree" or "Segment Tree for Lines")
        that supports:
          (1) adding a line of the form m*x + b,
          (2) querying the maximum value at a given x.
        
        The x in our problem is the index j, which ranges from 0 to n-1.
        The slope m = nums[i], and the intercept b = dp[i] - i*nums[i].
        
        Algorithm sketch:
          dp[0] = 0  (starting at index 0 with score 0)
          Add the line corresponding to i=0:
              slope = nums[0], intercept = dp[0] - 0*nums[0] = 0
          For i in 1..(n-1):
              dp[i] = query(i)  (the maximum value among all lines at x=i)
              Add line for i -> slope = nums[i], intercept = dp[i] - i*nums[i].
          Return dp[n-1].
        
        Complexity: O(n log X) where X ~ n if we use a Li Chao Tree over [0..n].
        This fits the constraints for n up to 1e5.
        """

        class LiChaoTree:
            """
            A Li Chao Tree for maximum lines over the domain [left_bound, right_bound].
            
            Each node keeps a 'best line' that's maximally dominant for some portion
            of the segment. We insert lines one by one, and for queries we find the
            max value at a given x.
            """
            INF = 10**18

            def __init__(self, left, right):
                self.left = left
                self.right = right
                self.m = 0     # slope
                self.b = -self.INF  # intercept (large negative for initialization)
                self.left_child = None
                self.right_child = None

            def f(self, x):
                return self.m*x + self.b

            def add_line(self, m, b):
                """
                Add a new line (m*x + b) into the tree, maintaining the maximum.
                """
                # We'll do it recursively
                self._add_line(m, b, self.left, self.right)

            def _add_line(self, m, b, left, right):
                mid = (left + right) >> 1
                # Check current node line vs new line at mid
                current_mid_val = self.f(mid)
                new_mid_val = m*mid + b

                # If the new line is better at mid, we swap them
                if new_mid_val > current_mid_val:
                    # Swap the lines
                    self.m, m = m, self.m
                    self.b, b = b, self.b

                # Now we decide which side to recurse on
                if left == right:
                    return

                if m*mid + b > self.f(mid):
                    # If the old line is better on the left side, go left
                    if not self.left_child:
                        self.left_child = LiChaoTree(left, mid)
                        self.left_child.m = m
                        self.left_child.b = b
                    else:
                        self.left_child._add_line(m, b, left, mid)
                else:
                    # Otherwise go right
                    if not self.right_child:
                        self.right_child = LiChaoTree(mid+1, right)
                        self.right_child.m = m
                        self.right_child.b = b
                    else:
                        self.right_child._add_line(m, b, mid+1, right)

            def query(self, x):
                """
                Get the maximum value at x among all lines in the structure.
                """
                return self._query(x, self.left, self.right)

            def _query(self, x, left, right):
                res = self.f(x)
                if left == right:
                    return res

                mid = (left + right) >> 1
                if x <= mid and self.left_child:
                    return max(res, self.left_child._query(x, left, mid))
                elif x > mid and self.right_child:
                    return max(res, self.right_child._query(x, mid+1, right))
                return res

        n = len(nums)
        # Edge case: if there's only one element, answer is 0 (start = end)
        if n == 1:
            return 0

        dp = [0]*n
        dp[0] = 0  # starting score

        # Build Li Chao Tree over x in [0..n-1]
        tree = LiChaoTree(0, n-1)
        # Add line for i = 0
        tree.add_line(nums[0], dp[0] - 0*nums[0])  # slope = nums[0], intercept = 0

        for i in range(1, n):
            # dp[i] = best among all lines at x = i
            dp[i] = tree.query(i)
            # Now add line for i so it can be used for future j > i
            tree.add_line(nums[i], dp[i] - i*nums[i])

        return dp[n-1]
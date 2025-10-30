class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        """
        We want to maximize the total score when jumping from index 0 to index n-1.
        Score of jump from i to j (where j > i) is: (j - i) * nums[i].

        Define dp[j] = the maximum possible total score upon reaching index j.
        Then:
            dp[j] = max over i < j of [ dp[i] + (j - i)*nums[i] ]
                   = max over i < j of [ dp[i] - i*nums[i] + j*nums[i] ]

        If we let line(i): y = m*x + b, where m = nums[i], b = (dp[i] - i*nums[i]),
        then dp[j] = max( line(i).evaluate( x = j ) ) for i < j.

        We can maintain these lines and do queries "what is the maximum slope*x + intercept
        at x=j?" using a Li Chao tree. After computing dp[j], we add the line for index j.

        Time complexity: O(n log N), where n = len(nums) and N ~ range of x (up to n).
        """

        # -------------------------
        # Implementation of a Li Chao tree for maximum queries
        # -------------------------
        import sys
        sys.setrecursionlimit(10**7)

        class Line:
            __slots__ = ('m', 'b')
            def __init__(self, slope, intercept):
                self.m = slope
                self.b = intercept
            
            # Evaluate this line at x
            def __call__(self, x):
                return self.m * x + self.b

        class LiChaoTree:
            """
            Li Chao tree to maintain a set of lines (m*x + b) and answer max queries for x in [left, right].
            """
            __slots__ = ('left', 'right', 'm', 'line', 'lc', 'rc')
            def __init__(self, left, right, line=None):
                self.left = left
                self.right = right
                self.m = (left + right) // 2
                self.line = line   # the line that is currently best for the entire segment [left, right]
                self.lc = None     # left child
                self.rc = None     # right child

            def get_mid(self):
                return self.m

            def eval(self, line, x):
                return line(x) if line else float('-inf')

            def add_line(self, new_line):
                """
                Add a new line (because we want maximum).
                We'll compare it with the current self.line and decide
                where it is better, then recursively add to children as needed.
                """
                self._add_line(new_line, self.left, self.right)

            def _add_line(self, new_line, left, right):
                if self.line is None:
                    # If there's no line stored, store new_line here and return
                    self.line = new_line
                    return

                mid = self.m
                # Compare the two lines at mid
                if self.eval(new_line, mid) > self.eval(self.line, mid):
                    # new_line is better at x=mid, swap them
                    self.line, new_line = new_line, self.line

                # Now we know self.line is the best line in x=mid.
                # We check intervals to potentially store the other line better in a sub-range.
                if left == right:
                    # No sub-range to go
                    return

                if self.eval(new_line, left) > self.eval(self.line, left):
                    # If new_line is better at left, we add it into the left child
                    if not self.lc:
                        self.lc = LiChaoTree(left, mid)
                    self.lc._add_line(new_line, left, mid)
                if self.eval(new_line, right) > self.eval(self.line, right):
                    # If new_line is better at right, we add it into the right child
                    if not self.rc:
                        self.rc = LiChaoTree(mid+1, right)
                    self.rc._add_line(new_line, mid+1, right)

            def query(self, x):
                """
                Returns the maximum value among all lines stored, evaluated at x.
                """
                val = self.eval(self.line, x)
                if x <= self.m:
                    # query left side
                    if self.lc:
                        val = max(val, self.lc.query(x))
                else:
                    # query right side
                    if self.rc:
                        val = max(val, self.rc.query(x))
                return val

        n = len(nums)
        if n == 1:
            return 0  # If there's only one index, no jump is made -> score is 0

        # dp[j] = maximum score to reach j
        dp0 = 0  # dp[0] = 0
        # Build Li Chao Tree over x range [0, n-1] (that covers all possible index queries).
        lichao = LiChaoTree(0, n-1)

        # Add line for i = 0
        # slope = nums[0], intercept = dp[0] - 0 * nums[0] = 0
        lichao.add_line(Line(nums[0], dp0 - 0 * nums[0]))

        dp = dp0
        for j in range(1, n):
            # dp[j] = max over i<j of dp[i] + (j - i)*nums[i]
            #       = max over i<j of [ dp[i] - i*nums[i] + j*nums[i] ]
            dp_j = lichao.query(j)
            # Add line for index j: slope = nums[j], intercept = dp_j - j*nums[j]
            lichao.add_line(Line(nums[j], dp_j - j * nums[j]))
            dp = dp_j  # Keep track of dp[j] as we go

        return dp
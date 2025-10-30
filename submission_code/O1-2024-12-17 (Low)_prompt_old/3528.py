class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        """
        We want to maximize the total score to reach index n-1 from index 0.
        
        Let dp[i] = maximum total score upon reaching index i.

        From index i, a jump to index j (j > i) adds (j - i) * nums[i].
        So the transition is:
            dp[j] = max(dp[j], dp[i] + (j - i) * nums[i])
                  = max(dp[j], dp[i] + j*nums[i] - i*nums[i])
                  = max(dp[j], (dp[i] - i*nums[i]) + (nums[i] * j))

        Define:
            A[i] = dp[i] - i * nums[i]
            slope[i] = nums[i]
        
        Then:
            dp[j] = max over i<j of [A[i] + slope[i]*j]

        Observing that for each i, we have a "line" of the form:
            f_i(x) = slope[i] * x + A[i]
        and we want to find:  dp[j] = max_i f_i(j).

        We'll use a Li Chao Tree (or "Segment Tree for lines") to maintain
        the maximum of these lines and query efficiently as j increases.

        Steps:
         1. dp[0] = 0 (start with no score at index 0).
         2. Insert line corresponding to i=0:
            slope = nums[0], intercept = dp[0] - 0*nums[0] = 0.
         3. For j in [1..n-1]:
               dp[j] = query the Li Chao Tree at x=j
               dp[j] gives us the max of slope*i * j + A[i]
            Then add line for index j:
               slope = nums[j]
               intercept = dp[j] - j*nums[j]
         4. Answer is dp[n-1].

        Time complexity: O(n log X), where X is the query range (here up to n).
        Since n <= 10^5, this approach is feasible.
        """

        #---------------------------
        # Li Chao Tree implementation
        #---------------------------
        class LiChaoTree:
            # We'll build a dynamic Li Chao tree for x in [0, n-1].
            # Each node covers some segment [start, end].
            # We'll store "line" (m, b) and children.
            # We use a recursive structure with lazy insertion.

            def __init__(self, start, end):
                self.start = start
                self.end = end
                self.mid = (start + end) // 2
                self.line = None
                self.left = None
                self.right = None

            def f(self, line, x):
                m, b = line
                return m * x + b

            def add_line(self, new_line):
                """
                Insert new_line = (m, b) into the tree.
                We'll maintain the node's best line for any x in [start, end].
                """
                # If there's no line yet, just store and return
                if self.line is None:
                    self.line = new_line
                    return

                # Compare values at the midpoint
                current = self.line
                l, r = self.start, self.end
                m = self.mid

                # Evaluate current line vs new line at midpoint
                if self.f(new_line, m) > self.f(current, m):
                    # Swap so that the node always has the better line at 'm'
                    self.line, new_line = new_line, self.line

                # If they differ on the left half, go left
                if l == r:
                    return
                if self.f(new_line, l) > self.f(self.line, l):
                    if not self.left:
                        self.left = LiChaoTree(l, m)
                    self.left.add_line(new_line)
                # If they differ on the right half, go right
                if self.f(new_line, r) > self.f(self.line, r):
                    if not self.right:
                        self.right = LiChaoTree(m+1, r)
                    self.right.add_line(new_line)

            def query(self, x):
                """
                Get the maximum y-value for x within [start, end].
                """
                val = self.f(self.line, x) if self.line else float('-inf')
                if self.start == self.end or not self.line:
                    return val
                m = self.mid
                if x <= m and self.left:
                    return max(val, self.left.query(x))
                elif x > m and self.right:
                    return max(val, self.right.query(x))
                return val

        n = len(nums)
        # Edge case: if there's only one index, score is 0
        if n == 1:
            return 0

        # Initialize dp array
        dp = [0]*n
        dp[0] = 0

        # Build Li Chao Tree over x-range [0, n-1]
        tree = LiChaoTree(0, n-1)

        # Insert line for i=0
        # slope = nums[0], intercept = dp[0] - 0*nums[0] = 0
        tree.add_line((nums[0], dp[0] - 0 * nums[0]))

        for j in range(1, n):
            # Query the best line at x=j
            dp[j] = tree.query(j)
            # Now add the line corresponding to index j
            # slope = nums[j], intercept = dp[j] - j*nums[j]
            tree.add_line((nums[j], dp[j] - j * nums[j]))

        return dp[-1]
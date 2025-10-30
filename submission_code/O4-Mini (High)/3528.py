from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        # If there's only one element, no jumps can be made
        if n == 1:
            return 0

        # We'll build a Li Chao segment tree over x in [0, n-1]
        N = n - 1
        size = 4 * (N + 1)
        INF_NEG = -(10**30)

        # tree_m[node], tree_b[node] represent the line y = m*x + b stored at this node
        tree_m = [0] * size
        tree_b = [INF_NEG] * size

        # Insert the line corresponding to starting index 0:
        # slope = nums[0], intercept = dp[0] - 0*nums[0] = 0
        def add_line(m: int, b: int) -> None:
            node = 1
            l, r = 0, N
            while True:
                mid = (l + r) // 2
                # Compare new line f with stored line g at x = mid
                # f(mid) = m*mid + b, g(mid) = tree_m[node]*mid + tree_b[node]
                if m * mid + b > tree_m[node] * mid + tree_b[node]:
                    # Swap f and g so that tree stores the better line at mid
                    tree_m[node], m = m, tree_m[node]
                    tree_b[node], b = b, tree_b[node]
                # If we've reached a leaf, stop
                if l == r:
                    break
                # Decide which side the "worse" line might dominate
                if m * l + b > tree_m[node] * l + tree_b[node]:
                    # new line is better on the left endpoint
                    node = node * 2
                    r = mid
                elif m * r + b > tree_m[node] * r + tree_b[node]:
                    # new line is better on the right endpoint
                    node = node * 2 + 1
                    l = mid + 1
                else:
                    # nowhere better, stop inserting
                    break

        def query(x: int) -> int:
            node = 1
            l, r = 0, N
            best = INF_NEG
            while True:
                # Evaluate the line at this node
                val = tree_m[node] * x + tree_b[node]
                if val > best:
                    best = val
                if l == r:
                    break
                mid = (l + r) // 2
                if x <= mid:
                    node = node * 2
                    r = mid
                else:
                    node = node * 2 + 1
                    l = mid + 1
            return best

        # Initialize with the line for index 0
        add_line(nums[0], 0)

        # Iterate j from 1 to n-1, compute dp[j], insert its line (unless it's the last index)
        dp_val = 0
        for j in range(1, n):
            # dp[j] = max over i<j of (dp[i] + (j-i)*nums[i]) = query(j)
            dp_val = query(j)
            # prepare line for this j: slope = nums[j], intercept = dp[j] - j*nums[j]
            if j < n - 1:
                intercept = dp_val - j * nums[j]
                add_line(nums[j], intercept)

        # dp_val now holds dp[n-1]
        return dp_val
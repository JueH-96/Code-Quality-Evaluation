from typing import List

class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def value(self, x):
        return self.m * x + self.b

class LiChaoNode:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.line = None

    def add_line(self, new_line):
        if self.line is None:
            self.line = new_line
            return
        m = (self.l + self.r) // 2
        current_val = self.line.value(m)
        new_val = new_line.value(m)
        if new_val > current_val:
            temp = self.line
            self.line = new_line
            new_line = temp
        current_left = self.line.value(self.l)
        new_left = new_line.value(self.l)
        if new_left > current_left:
            if self.left is None:
                self.left = LiChaoNode(self.l, m)
            self.left.add_line(new_line)
        current_right = self.line.value(self.r)
        new_right = new_line.value(self.r)
        if new_right > current_right:
            if self.right is None:
                self.right = LiChaoNode(m + 1, self.r)
            self.right.add_line(new_line)

    def query_max(self, x):
        res = self.line.value(x) if self.line is not None else -float('inf')
        m = (self.l + self.r) // 2
        if x <= m and self.left is not None:
            left_res = self.left.query_max(x)
            res = max(res, left_res)
        elif x > m and self.right is not None:
            right_res = self.right.query_max(x)
            res = max(res, right_res)
        return res

class LiChaoTree:
    def __init__(self, x_min, x_max):
        self.root = LiChaoNode(x_min, x_max)

    def add_line(self, m, b):
        new_line = Line(m, b)
        self.root.add_line(new_line)

    def query(self, x):
        return self.root.query_max(x)

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Initialize Li Chao Tree with the range of possible x values
        # The maximum x can be up to 1e5 as per the problem constraints
        max_x = 10**5
        tree = LiChaoTree(0, max_x)
        dp = [0] * n
        for i in range(n-1, -1, -1):
            if i == n-1:
                dp[i] = 0
            else:
                a = nums[i]
                current_max = tree.query(a)
                dp[i] = current_max - a * i
            # Add the line for the current i
            m = i
            b = dp[i]
            tree.add_line(m, b)
        return dp[0]
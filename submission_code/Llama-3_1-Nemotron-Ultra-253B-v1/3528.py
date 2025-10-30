from typing import List

class Line:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get(self, x):
        return self.a * x + self.b

class LiChaoNode:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.line = None

class LiChaoTree:
    def __init__(self, x_min, x_max):
        self.root = LiChaoNode(x_min, x_max)

    def add_line(self, new_line):
        stack = [(self.root, new_line)]
        while stack:
            node, line = stack.pop()
            l = node.l
            r = node.r
            m = (l + r) // 2
            if node.line is None:
                node.line = line
                continue
            curr_val = node.line.get(m)
            new_val = line.get(m)
            if new_val > curr_val:
                node.line, line = line, node.line
            curr_val_l = node.line.get(l)
            new_val_l = line.get(l)
            if new_val_l > curr_val_l:
                if not node.left:
                    node.left = LiChaoNode(l, m)
                stack.append((node.left, line))
            curr_val_r = node.line.get(r)
            new_val_r = line.get(r)
            if new_val_r > curr_val_r:
                if not node.right:
                    node.right = LiChaoNode(m + 1, r)
                stack.append((node.right, line))

    def query(self, x):
        res = -float('inf')
        node = self.root
        while node:
            if node.line:
                res = max(res, node.line.get(x))
            m = (node.l + node.r) // 2
            if x <= m:
                node = node.left
            else:
                node = node.right
        return res

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        x_max = n - 1
        tree = LiChaoTree(0, x_max)
        dp = [0] * n
        tree.add_line(Line(nums[0], 0))
        for i in range(1, n):
            dp[i] = tree.query(i)
            a = nums[i]
            b = dp[i] - i * a
            tree.add_line(Line(a, b))
        return dp[-1]
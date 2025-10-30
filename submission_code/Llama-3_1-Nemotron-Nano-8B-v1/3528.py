from typing import List

class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def get(self, x):
        return self.m * x + self.b

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
            current_val = node.line.get(m)
            new_val = line.get(m)
            if new_val > current_val:
                node.line, line = line, node.line
            current_val_l = node.line.get(l)
            new_val_l = line.get(l)
            current_val_r = node.line.get(r)
            new_val_r = line.get(r)
            if new_val_r > current_val_r:
                if not node.right:
                    node.right = LiChaoNode(m + 1, r)
                stack.append((node.right, line))
            if new_val_l > current_val_l:
                if not node.left:
                    node.left = LiChaoNode(l, m)
                stack.append((node.left, line))

    def query(self, x):
        res = -float('inf')
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.line:
                res = max(res, node.line.get(x))
            m = (node.l + node.r) // 2
            if x <= m and node.left:
                stack.append(node.left)
            elif x > m and node.right:
                stack.append(node.right)
        return res

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        x_min = 1
        x_max = 10**5
        tree = LiChaoTree(x_min, x_max)
        m_initial = nums[0]
        b_initial = 0 - 0 * m_initial
        tree.add_line(Line(m_initial, b_initial))
        dp = [0] * n
        dp[0] = 0
        for i in range(1, n):
            current_max = tree.query(i)
            dp[i] = current_max
            m_i = nums[i]
            b_i = dp[i] - i * m_i
            tree.add_line(Line(m_i, b_i))
        return dp[-1]
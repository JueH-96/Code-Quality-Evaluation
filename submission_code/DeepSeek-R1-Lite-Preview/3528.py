from typing import List

class LiChaoTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.lines = [None] * (2 * self.size)
    
    def insert(self, left, right, line, node=1, node_left=0, node_right=None):
        if node_right is None:
            node_right = self.size - 1
        if right < node_left or left > node_right:
            return
        if left <= node_left and node_right <= right:
            existing = self.lines[node]
            if existing is None:
                self.lines[node] = line
                return
            m1, c1 = existing
            m2, c2 = line
            if m1 == m2:
                if c2 > c1:
                    self.lines[node] = line
                return
            x_intersect = (c1 - c2) / (m2 - m1)
            if x_intersect > (node_left + node_right) / 2:
                if c2 + m2 * node_left > c1 + m1 * node_left:
                    self.lines[node], line = line, self.lines[node]
                self.insert(left, right, line, node * 2 + 1, (node_left + node_right) // 2 + 1, node_right)
            else:
                if c2 + m2 * node_right > c1 + m1 * node_right:
                    self.lines[node], line = line, self.lines[node]
                self.insert(left, right, line, node * 2, node_left, (node_left + node_right) // 2)
            return
        self.insert(left, right, line, node * 2, node_left, (node_left + node_right) // 2)
        self.insert(left, right, line, node * 2 + 1, (node_left + node_right) // 2 + 1, node_right)
    
    def query(self, x, node=1, node_left=0, node_right=None):
        if node_right is None:
            node_right = self.size - 1
        if x < node_left or x > node_right:
            return float('-inf')
        y = float('-inf')
        if self.lines[node] is not None:
            m, c = self.lines[node]
            y = m * x + c
        if node_left == node_right:
            return y
        mid = (node_left + node_right) // 2
        if x <= mid:
            return max(y, self.query(x, node * 2, node_left, mid))
        else:
            return max(y, self.query(x, node * 2 + 1, mid + 1, node_right))

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        lichao = LiChaoTree(n)
        lichao.insert(0, n - 1, (nums[0], dp[0] - 0 * nums[0]))
        for i in range(1, n):
            dp[i] = lichao.query(i)
            lichao.insert(0, n - 1, (nums[i], dp[i] - i * nums[i]))
        return dp[n - 1]
from typing import List

class Line:
    def __init__(self, m=0, b=float('-inf')):
        self.m = m  # slope
        self.b = b  # intercept

    def value_at(self, x):
        return self.m * x + self.b

    def intersect_x(self, other):
        return (other.b - self.b) / (self.m - other.m)

class ConvexHullTrick:
    def __init__(self):
        self.lines = []
        self.pointer = 0

    def add_line(self, m, b):
        new = Line(m, b)
        while len(self.lines) >= 2:
            l1 = self.lines[-2]
            l2 = self.lines[-1]
            # If intersection of l1 and l2 is >= intersection of l2 and new, remove l2
            if (l1.intersect_x(l2)) >= (l2.intersect_x(new)):
                self.lines.pop()
            else:
                break
        self.lines.append(new)

    def query(self, x):
        while self.pointer +1 < len(self.lines) and self.lines[self.pointer +1].value_at(x) >= self.lines[self.pointer].value_at(x):
            self.pointer +=1
        return self.lines[self.pointer].value_at(x)

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        cht = ConvexHullTrick()
        # Initialize with the first line
        cht.add_line(nums[0], dp[0] - nums[0]*0)
        for j in range(1, n):
            dp[j] = cht.query(j)  # dp[j] = max_i (dp[i] + (j - i)*nums[i]) = max_i (nums[i]*j + dp[i] - nums[i]*i)
            if j == n -1:
                break
            # Add the line for current index
            cht.add_line(nums[j], dp[j] - nums[j]*j)
        return dp[-1]
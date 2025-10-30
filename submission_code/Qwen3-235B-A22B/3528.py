from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        next_ge = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                prev = stack.pop()
                next_ge[prev] = i
            stack.append(i)
        
        total = 0
        current = 0
        while current < n - 1:
            if next_ge[current] == -1:
                total += (n - 1 - current) * nums[current]
                break
            else:
                total += (next_ge[current] - current) * nums[current]
                current = next_ge[current]
        return total
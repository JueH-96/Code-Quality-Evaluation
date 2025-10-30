from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        stack = []
        max_score = 0
        
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                mid = stack.pop()
                if stack:
                    left = stack[-1]
                else:
                    left = -1
                score = (i - left - 1) * nums[mid]
                max_score = max(max_score, score)
            stack.append(i)
        
        while stack:
            mid = stack.pop()
            if stack:
                left = stack[-1]
            else:
                left = -1
            score = (n - left - 1) * nums[mid]
            max_score = max(max_score, score)
        
        return max_score
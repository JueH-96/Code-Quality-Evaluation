from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        # Compute next greater index to the right
        ng = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                ng[idx] = i
            stack.append(i)
        
        # Simulate the jumps using greedy approach
        score = 0
        curr = 0
        while curr < n - 1:
            if ng[curr] < n:
                next_pos = ng[curr]
            else:
                next_pos = n - 1
            score += (next_pos - curr) * nums[curr]
            curr = next_pos
        
        return score
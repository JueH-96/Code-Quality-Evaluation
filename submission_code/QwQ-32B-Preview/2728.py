from collections import deque
from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        # Convert each row to deque for efficient popleft
        deques = [deque(row) for row in nums]
        
        score = 0
        
        while True:
            # Collect the first element from each deque that still has elements
            removed = [dq.popleft() for dq in deques if dq]
            if not removed:
                break  # No more elements to remove
            # Add the maximum of the removed elements to the score
            score += max(removed)
        
        return score
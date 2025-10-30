from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Preprocess the positions of each number
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        left, right = 0, len(nums)
        answer = 0
        
        # Binary search on the possible maximum count
        while left <= right:
            mid = (left + right) // 2
            if self.is_possible(mid, pos, k, nums):
                answer = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return answer
    
    def is_possible(self, c: int, pos: dict, k: int, nums: List[int]) -> bool:
        if c == 0:
            return True  # Empty subarray is always possible
        
        for num in pos:
            positions = pos[num]
            m = len(positions)
            if m < c:
                continue  # Not enough occurrences of this number
            
            j = 0
            for i in range(m):
                # Find the smallest j for current i such that window is valid
                while j <= i and (i - j + 1 > c or (positions[i] - positions[j] + 1) > (c + k)):
                    j += 1
                if i - j + 1 >= c:
                    return True
        
        return False
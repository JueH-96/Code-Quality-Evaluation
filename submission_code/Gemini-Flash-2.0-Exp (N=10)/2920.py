from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        first_occurrence = {}
        last_occurrence = {}
        
        for i, num in enumerate(nums):
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        if len(first_occurrence) == 1:
            return 0
        
        max_dist = 0
        for num in first_occurrence:
            first = first_occurrence[num]
            last = last_occurrence[num]
            
            
            
            indices = []
            for i, val in enumerate(nums):
                if val == num:
                    indices.append(i)
            
            
            
            
            max_gap = 0
            for i in range(len(indices)):
                
                
                gap = 0
                if i == len(indices) - 1:
                    gap = (indices[0] + n - indices[i])
                else:
                    gap = indices[i+1] - indices[i]
                
                max_gap = max(max_gap, gap)
            
            max_dist = max(max_dist, (max_gap + 1) // 2)
            
        return max_dist
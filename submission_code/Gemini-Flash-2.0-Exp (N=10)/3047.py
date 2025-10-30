import math
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = 0
        
        def is_perfect_square(n):
            if n < 0:
                return False
            root = int(math.sqrt(n))
            return root * root == n
        
        def is_complete(subset_indices):
            if len(subset_indices) <= 1:
                return True
            
            for i in range(len(subset_indices)):
                for j in range(i + 1, len(subset_indices)):
                    prod = nums[subset_indices[i]-1] * nums[subset_indices[j]-1]
                    if not is_perfect_square(prod):
                        return False
            return True
        
        for i in range(1, 1 << n):
            subset_indices = []
            for j in range(n):
                if (i >> j) & 1:
                    subset_indices.append(j+1)
            
            if is_complete(subset_indices):
                current_sum = 0
                for index in subset_indices:
                    current_sum += nums[index-1]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
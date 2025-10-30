from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        max_outlier = -float('inf')

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                special_nums = []
                sum_val = -float('inf')
                outlier = -float('inf')
                
                
                for k in range(n):
                    if k != i and k != j:
                        special_nums.append(nums[k])
                
                
                if len(special_nums) == n - 2:
                    sum_special = sum(special_nums)
                    if nums[i] == sum_special:
                        outlier = nums[j]
                    elif nums[j] == sum_special:
                        outlier = nums[i]
                    
                    if outlier != -float('inf'):
                        max_outlier = max(max_outlier, outlier)
        
        return max_outlier
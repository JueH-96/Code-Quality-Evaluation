from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        found_mountain = False

        for j in range(1, n-1):
            # Find the smallest nums[i] such that i < j and nums[i] < nums[j]
            min_left = float('inf')
            for i in range(j):
                if nums[i] < nums[j] and nums[i] < min_left:
                    min_left = nums[i]
            
            # Find the smallest nums[k] such that k > j and nums[k] < nums[j]
            min_right = float('inf')
            for k in range(j+1, n):
                if nums[k] < nums[j] and nums[k] < min_right:
                    min_right = nums[k]
            
            # If both sides found valid values, update the minimum sum
            if min_left != float('inf') and min_right != float('inf'):
                found_mountain = True
                current_sum = min_left + nums[j] + min_right
                if current_sum < min_sum:
                    min_sum = current_sum
        
        return min_sum if found_mountain else -1
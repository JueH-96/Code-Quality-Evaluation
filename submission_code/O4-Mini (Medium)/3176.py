from typing import List
import math

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        best_sum = math.inf
        
        for j in range(1, n-1):
            # find the smallest nums[i] to the left of j that is < nums[j]
            left_min = math.inf
            for i in range(j):
                if nums[i] < nums[j] and nums[i] < left_min:
                    left_min = nums[i]
            
            # find the smallest nums[k] to the right of j that is < nums[j]
            right_min = math.inf
            for k in range(j+1, n):
                if nums[k] < nums[j] and nums[k] < right_min:
                    right_min = nums[k]
            
            # if both sides have valid candidates, update the best sum
            if left_min < math.inf and right_min < math.inf:
                current_sum = left_min + nums[j] + right_min
                if current_sum < best_sum:
                    best_sum = current_sum
        
        return best_sum if best_sum < math.inf else -1

# Example usage:
# sol = Solution()
# print(sol.minimumSum([8,6,1,5,3]))  # Output: 9
# print(sol.minimumSum([5,4,8,7,10,2]))  # Output: 13
# print(sol.minimumSum([6,5,4,3,4,5]))  # Output: -1
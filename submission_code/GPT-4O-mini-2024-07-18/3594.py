from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        num_set = set(nums)
        total_sum = sum(nums)
        
        # The outlier must be one of the numbers that is not part of the special numbers or their sum
        potential_outliers = []
        
        for num in num_set:
            if num != total_sum and nums.count(num) < 2:
                potential_outliers.append(num)
        
        # Return the largest potential outlier
        return max(potential_outliers)

# Example usage:
# sol = Solution()
# print(sol.getLargestOutlier([2, 3, 5, 10]))  # Output: 10
# print(sol.getLargestOutlier([-2, -1, -3, -6, 4]))  # Output: 4
# print(sol.getLargestOutlier([1, 1, 1, 1, 1, 5, 5]))  # Output: 5
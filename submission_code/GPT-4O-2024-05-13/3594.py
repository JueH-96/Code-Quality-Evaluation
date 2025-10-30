from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        n = len(nums)
        
        # Calculate the sum of all elements except each element
        potential_sums = [total_sum - nums[i] for i in range(n)]
        
        # Find the sum element
        sum_element = None
        for s in potential_sums:
            if (total_sum - s) in nums:
                sum_element = total_sum - s
                break
        
        # Remove the sum element from the list
        nums.remove(sum_element)
        
        # Calculate the sum of the remaining elements
        remaining_sum = sum(nums)
        
        # The outlier will be the element that is not part of the sum
        outlier = None
        for num in nums:
            if remaining_sum - num != sum_element:
                outlier = num
                break
        
        return max(outlier, sum_element)

# Example usage:
# sol = Solution()
# print(sol.getLargestOutlier([2, 3, 5, 10]))  # Output: 10
# print(sol.getLargestOutlier([-2, -1, -3, -6, 4]))  # Output: 4
# print(sol.getLargestOutlier([1, 1, 1, 1, 1, 5, 5]))  # Output: 5
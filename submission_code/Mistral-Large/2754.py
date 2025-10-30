from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # Sort the array
        nums.sort()

        # Initialize variables to store the maximum product
        max_product = float('-inf')
        n = len(nums)

        # Helper function to calculate the product of a subset
        def calculate_product(subset):
            product = 1
            for num in subset:
                product *= num
            return product

        # Consider all positive numbers
        positive_nums = [num for num in nums if num > 0]
        if positive_nums:
            max_product = max(max_product, calculate_product(positive_nums))

        # Consider all numbers (including negatives and zeroes)
        max_product = max(max_product, calculate_product(nums))

        # Consider the product of all positive numbers and the two largest negative numbers
        if len(nums) >= 2 and nums[0] < 0 and nums[1] < 0:
            max_product = max(max_product, calculate_product(positive_nums + nums[:2]))

        return max_product

# Example usage:
# solution = Solution()
# print(solution.maxStrength([3,-1,-5,2,5,-9]))  # Output: 1350
# print(solution.maxStrength([-4,-5,-4]))        # Output: 20
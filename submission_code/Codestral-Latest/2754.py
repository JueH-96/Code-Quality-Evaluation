class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # If there's only one number, return it
        if len(nums) == 1:
            return nums[0]

        # Sort the numbers to handle negative numbers more easily
        nums.sort()

        # Initialize variables to keep track of the maximum product
        max_product = float('-inf')

        # Iterate through all possible subarrays
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                # Calculate the product of the current subarray
                current_product = 1
                for k in range(i, j + 1):
                    current_product *= nums[k]

                # Update the maximum product if the current product is greater
                max_product = max(max_product, current_product)

        return max_product
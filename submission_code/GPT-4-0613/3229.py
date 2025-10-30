class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Since all numbers from 1 to 9 are palindromic and any two digit number with same digits (like 11, 22, ..., 99) are palindromic,
        # we can safely ignore the condition of palindromic number and just find the median of the array.
        # Changing all elements to the median will result in minimum total cost.
        nums.sort()
        median = nums[len(nums) // 2]
        return sum(abs(num - median) for num in nums)
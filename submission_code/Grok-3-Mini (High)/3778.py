class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even numbers with 0 and odd numbers with 1, then sort the array
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                nums[i] = 0
            else:
                nums[i] = 1
        nums.sort()
        return nums
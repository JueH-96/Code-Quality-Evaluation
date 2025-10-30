class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Initialize the sum of the longest sequential prefix
        seq_sum = nums[0]
        # Iterate through the nums array to find the longest sequential prefix
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                seq_sum += nums[i]
            else:
                break
        # Find the smallest missing integer greater than or equal to seq_sum
        x = seq_sum
        while x in nums:
            x += 1
        return x
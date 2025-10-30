class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums) - 2):
            # Check if 2 * (first + third) equals the middle element
            if 2 * (nums[i] + nums[i + 2]) == nums[i + 1]:
                count += 1
        return count
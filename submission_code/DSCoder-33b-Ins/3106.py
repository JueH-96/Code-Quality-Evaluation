class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        max_length = -1
        while left <= right:
            if nums[left] + nums[right] > target:
                right -= 1
            else:
                max_length = max(max_length, right - left + 1)
                left += 1
        return max_length if max_length != -1 else -1
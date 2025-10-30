class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        total = 0
        current_run = 1
        total += current_run
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current_run += 1
            else:
                current_run = 1
            total += current_run
        return total
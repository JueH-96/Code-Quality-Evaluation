class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        total = 0
        current_length = 1
        total += current_length  # Account for the first element
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current_length += 1
            else:
                current_length = 1
            total += current_length
        return total
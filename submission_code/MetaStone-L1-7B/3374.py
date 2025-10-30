class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        total = 0
        current_length = 1
        for i in range(len(nums)):
            if i > 0:
                if nums[i] != nums[i - 1]:
                    current_length += 1
                else:
                    current_length = 1
            total += current_length
        return total
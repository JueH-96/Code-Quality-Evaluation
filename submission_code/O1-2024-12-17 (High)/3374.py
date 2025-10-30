class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        length = 0  # Length of current alternating segment
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                length += 1
            else:
                length = 1
            count += length
        return count
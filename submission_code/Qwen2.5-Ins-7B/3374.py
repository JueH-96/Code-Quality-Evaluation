class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count, length = 0, 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                length += 1
            else:
                count += length * (length + 1) // 2
                length = 1
        count += length * (length + 1) // 2
        return count
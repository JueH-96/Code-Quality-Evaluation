class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        and_val = -1
        count = 0
        for num in nums:
            if and_val == -1:
                and_val = num
            else:
                and_val &= num
            if and_val == 0:
                count += 1
                and_val = -1
        return max(1, count)
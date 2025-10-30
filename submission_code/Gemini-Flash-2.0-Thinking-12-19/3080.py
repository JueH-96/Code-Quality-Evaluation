class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 0
        current_and = -1
        for num in nums:
            if current_and == -1:
                current_and = num
            else:
                current_and &= num
            if current_and == 0:
                count += 1
                current_and = -1
        return max(1, count)
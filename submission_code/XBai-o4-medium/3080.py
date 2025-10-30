class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        current_and = None
        split_count = 0
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            if current_and == 0:
                split_count += 1
                current_and = None
        return split_count if split_count else 1
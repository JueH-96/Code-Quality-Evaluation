class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        max_count = max(Counter(nums).values())
        return max(0, n - 2 * max_count + 1)
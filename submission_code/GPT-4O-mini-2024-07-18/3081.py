class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        max_freq = max(count.values())
        return len(nums) - 2 * max_freq if len(nums) >= 2 * max_freq else len(nums) % 2
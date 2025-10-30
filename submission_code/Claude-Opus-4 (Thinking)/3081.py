class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        max_freq = max(Counter(nums).values())
        
        return max(n % 2, 2 * max_freq - n)
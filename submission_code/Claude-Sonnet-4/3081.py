class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each element
        counter = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(counter.values())
        
        n = len(nums)
        
        # If the most frequent element appears more than half the time,
        # we'll have leftover elements
        if max_freq > n // 2:
            return 2 * max_freq - n
        else:
            return n % 2
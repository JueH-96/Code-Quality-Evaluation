class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Calculate the minimum length after removals
        n = len(nums)
        if max_freq > n // 2:
            return 2 * max_freq - n
        else:
            return n % 2
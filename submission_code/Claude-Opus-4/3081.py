class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Count frequency of each element
        from collections import Counter
        freq = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # If the most frequent element appears more than n/2 times,
        # we'll have at least (2 * max_freq - n) elements left
        if max_freq > n // 2:
            return 2 * max_freq - n
        
        # Otherwise, we can pair up elements optimally
        # If n is even, we can remove all elements
        # If n is odd, we'll have 1 element left
        return n % 2
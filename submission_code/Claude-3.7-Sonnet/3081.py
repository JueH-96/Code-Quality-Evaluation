class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the maximum frequency of any number
        max_freq = 1
        current_freq = 1
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                current_freq += 1
            else:
                max_freq = max(max_freq, current_freq)
                current_freq = 1
        max_freq = max(max_freq, current_freq)  # Don't forget the last block
        
        # Case 1: max_freq is at most half the array length
        if max_freq <= n // 2:
            return n % 2  # 0 if n is even, 1 if n is odd
        # Case 2: max_freq is more than half the array length
        else:
            # We can only pair (n - max_freq) elements with the most frequent value
            # This leaves (max_freq - (n - max_freq)) = 2*max_freq - n elements
            return 2 * max_freq - n
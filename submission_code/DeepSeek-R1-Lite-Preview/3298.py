from typing import List
from collections import defaultdict

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
            freq[num + 1] += 1
        
        sorted_x = sorted(freq.keys())
        max_length = 0
        current_sum = 0
        left = 0
        
        for right in range(len(sorted_x)):
            current_sum += freq[sorted_x[right]]
            
            # While the current window doesn't consist of consecutive integers
            # or the sum of frequencies is less than the window length
            while sorted_x[right] - sorted_x[left] > right - left:
                current_sum -= freq[sorted_x[left]]
                left += 1
            
            # Update the maximum length if the sum of frequencies is at least the window length
            window_length = right - left + 1
            if current_sum >= window_length:
                max_length = max(max_length, window_length)
        
        return max_length
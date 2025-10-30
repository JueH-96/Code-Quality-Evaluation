from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element in the entire array
        total_count = Counter(nums)
        n = len(nums)
        
        # Find the dominant element
        dominant = None
        for num, count in total_count.items():
            if count * 2 > n:
                dominant = num
                break
        
        # If no dominant element is found, return -1 (though problem guarantees one exists)
        if dominant is None:
            return -1
        
        # Initialize counters for the left and right parts
        left_count = 0
        right_count = total_count[dominant]
        
        # Iterate through the array to find the minimum valid split index
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
                right_count -= 1
            
            # Check if both parts have the same dominant element
            if left_count * 2 > (i + 1) and right_count * 2 > (n - i - 1):
                return i
        
        return -1
from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        answer = []
        n = len(nums)
        
        for i in range(n - k + 1):
            window = nums[i:i + k]
            freq = Counter(window)
            
            # Sort by frequency descending, then by value descending
            sorted_freq = sorted(freq.items(), key=lambda item: (-item[1], -item[0]))
            
            # Get the top x elements
            top_x_elements = sorted_freq[:x]
            top_elements_set = set([element for element, count in top_x_elements])
            
            # Calculate the sum of elements in the window that are in the top x
            window_sum = sum(num for num in window if num in top_elements_set)
            answer.append(window_sum)
        
        return answer
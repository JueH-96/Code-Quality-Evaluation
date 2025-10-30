from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            freq_count = Counter(subarray)
            
            # If there are fewer than x distinct elements, x-sum is the sum of the entire subarray
            if len(freq_count) <= x:
                result.append(sum(subarray))
                continue
            
            # Sort elements based on frequency (primary) and value (secondary for ties)
            sorted_elements = sorted(freq_count.keys(), key=lambda elem: (freq_count[elem], elem), reverse=True)
            
            # Take top x most frequent elements
            top_x_elements = set(sorted_elements[:x])
            
            # Calculate sum of all occurrences of the top x most frequent elements
            x_sum = sum(elem for elem in subarray if elem in top_x_elements)
            
            result.append(x_sum)
            
        return result
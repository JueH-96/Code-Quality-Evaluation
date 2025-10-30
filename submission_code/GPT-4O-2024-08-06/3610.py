from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def calculate_x_sum(subarray: List[int], x: int) -> int:
            # Count occurrences of each element
            count = Counter(subarray)
            # Sort elements by frequency and then by value (descending)
            sorted_elements = sorted(count.items(), key=lambda item: (-item[1], -item[0]))
            # Keep only the top x most frequent elements
            top_elements = set(elem for elem, _ in sorted_elements[:x])
            # Calculate the sum of the resulting array
            return sum(elem for elem in subarray if elem in top_elements)
        
        n = len(nums)
        result = []
        
        # Iterate over each subarray of length k
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            x_sum = calculate_x_sum(subarray, x)
            result.append(x_sum)
        
        return result
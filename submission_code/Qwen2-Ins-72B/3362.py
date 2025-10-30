from typing import List
from sortedcontainers import SortedList

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        last_occurrence = {num: i for i, num in enumerate(nums)}
        sorted_unique = SortedList()
        current_unique = set()
        
        for i, num in enumerate(nums):
            if num not in current_unique:
                current_unique.add(num)
                sorted_unique.add(num)
            
            # Remove elements that won't appear in any subarray starting from i
            while sorted_unique and last_occurrence[sorted_unique[0]] < i:
                current_unique.remove(sorted_unique[0])
                sorted_unique.pop(0)
            
            # Calculate the median of the uniqueness array up to index i
            if (i + 1) % 2 == 0:
                median = (sorted_unique[len(sorted_unique) // 2 - 1] + sorted_unique[len(sorted_unique) // 2]) / 2
            else:
                median = sorted_unique[len(sorted_unique) // 2]
            
            # Update the answer with the current median
            if i == n // 2:
                return int(median)
        
        return int(sorted_unique[len(sorted_unique) // 2])
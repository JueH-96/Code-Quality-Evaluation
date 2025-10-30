from typing import List
from collections import Counter

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        # Count all elements as the initial suffix
        suffix_counts = Counter(nums)
        prefix_set = set()
        result = []
        
        for x in nums:
            # Remove current element from the suffix
            suffix_counts[x] -= 1
            if suffix_counts[x] == 0:
                del suffix_counts[x]
            
            # Add current element to the prefix set
            prefix_set.add(x)
            
            # Compute distinct counts
            prefix_distinct = len(prefix_set)
            suffix_distinct = len(suffix_counts)
            
            # Append the difference
            result.append(prefix_distinct - suffix_distinct)
        
        return result
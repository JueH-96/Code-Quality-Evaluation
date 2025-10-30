import collections
from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        """
        Calculates the distinct difference array using a single-pass O(n) approach.
        """
        
        # A set to keep track of distinct elements in the prefix as we iterate.
        prefix_set = set()
        
        # A frequency map for elements. Initially, it contains all elements,
        # representing the pool for suffix elements.
        suffix_counter = collections.Counter(nums)
        
        diff = []
        
        for num in nums:
            # The current element `num` is now part of the prefix `nums[0...i]`.
            prefix_set.add(num)
            
            # The current element is no longer part of the suffix `nums[i+1...n-1]`.
            suffix_counter[num] -= 1
            if suffix_counter[num] == 0:
                # If an element's count drops to zero, it's no longer a distinct
                # element in the suffix, so we remove its key from the counter.
                del suffix_counter[num]
            
            # The number of distinct elements is the size of the respective data structure.
            prefix_distinct_count = len(prefix_set)
            suffix_distinct_count = len(suffix_counter)
            
            # Calculate and store the difference for the current index.
            diff.append(prefix_distinct_count - suffix_distinct_count)
            
        return diff
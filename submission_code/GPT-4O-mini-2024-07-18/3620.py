from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Use a set to track distinct elements
        distinct_elements = set()
        
        # Sort the array to handle elements in order
        nums.sort()
        
        for num in nums:
            # If the number is already distinct, add it directly
            if num not in distinct_elements:
                distinct_elements.add(num)
            else:
                # If the number is not distinct, try to adjust it
                # Check if we can add k to make it distinct
                if (num + k) not in distinct_elements:
                    distinct_elements.add(num + k)
                # Check if we can subtract k to make it distinct
                elif (num - k) not in distinct_elements:
                    distinct_elements.add(num - k)
        
        # The size of the set is the maximum number of distinct elements
        return len(distinct_elements)
from typing import List
from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Count the frequency of each number in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)
        
        # Sort the counts in descending order
        counts1 = sorted(count1.values(), reverse=True)
        counts2 = sorted(count2.values(), reverse=True)
        
        # Initialize the maximum size of the set and the current size of the set
        max_size = 0
        current_size = 0
        
        # Initialize the indices for the counts of nums1 and nums2
        i = 0
        j = 0
        
        # While there are still counts to consider
        while i < len(counts1) or j < len(counts2):
            # If there are still counts in nums1 and nums2
            if i < len(counts1) and j < len(counts2):
                # If the count in nums1 is greater than or equal to the count in nums2
                if counts1[i] >= counts2[j]:
                    # Add the count to the current size of the set
                    current_size += counts1[i]
                    # Increment the index for nums1
                    i += 1
                else:
                    # Otherwise, add the count to the current size of the set
                    current_size += counts2[j]
                    # Increment the index for nums2
                    j += 1
            # If there are still counts in nums1
            elif i < len(counts1):
                # Add the count to the current size of the set
                current_size += counts1[i]
                # Increment the index for nums1
                i += 1
            # Otherwise, there are still counts in nums2
            else:
                # Add the count to the current size of the set
                current_size += counts2[j]
                # Increment the index for nums2
                j += 1
            
            # Update the maximum size of the set
            max_size = max(max_size, current_size)
        
        return max_size
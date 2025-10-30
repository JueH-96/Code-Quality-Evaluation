from typing import List
from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Count the frequency of each element in both arrays
        count1 = Counter(nums1)
        count2 = Counter(nums2)

        # Initialize a set to store unique elements
        unique_elements = set()

        # Add elements from both arrays to the set
        for num in count1:
            unique_elements.add(num)
        for num in count2:
            unique_elements.add(num)

        # Initialize the maximum possible size of the set
        max_size = 0

        # Iterate over all possible subsets of unique elements
        for mask in range(1 << len(unique_elements)):
            subset = set()
            for i, num in enumerate(unique_elements):
                if (mask & (1 << i)):
                    subset.add(num)

            # Check if it's possible to remove elements to get the current subset
            possible = True
            for num in subset:
                if count1[num] + count2[num] < len(subset):
                    possible = False
                    break

            # Update the maximum possible size of the set
            if possible:
                max_size = max(max_size, len(subset))

        return max_size
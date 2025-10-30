from typing import List

class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Create sets of unique elements from each list for efficient lookups.
        # Checking if an element is in a set is O(1) on average.
        set1 = set(nums1)
        set2 = set(nums2)

        # Calculate the first value:
        # Count the number of elements in nums1 that are present in set2 (derived from nums2).
        count1 = 0
        for num in nums1:
            if num in set2:
                count1 += 1

        # Calculate the second value:
        # Count the number of elements in nums2 that are present in set1 (derived from nums1).
        count2 = 0
        for num in nums2:
            if num in set1:
                count2 += 1

        # Return the two counts as a list.
        return [count1, count2]
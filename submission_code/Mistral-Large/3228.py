from typing import List
from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half_n = n // 2

        # Count the frequency of each element in nums1 and nums2
        count1 = Counter(nums1)
        count2 = Counter(nums2)

        # Sort the elements by their frequency in descending order
        sorted_count1 = sorted(count1.items(), key=lambda x: -x[1])
        sorted_count2 = sorted(count2.items(), key=lambda x: -x[1])

        # Remove the most frequent elements from nums1
        removed1 = set()
        remaining1 = half_n
        for num, freq in sorted_count1:
            if remaining1 <= 0:
                break
            if freq <= remaining1:
                remaining1 -= freq
                removed1.add(num)
            else:
                remaining1 = 0
                break

        # Remove the most frequent elements from nums2
        removed2 = set()
        remaining2 = half_n
        for num, freq in sorted_count2:
            if remaining2 <= 0:
                break
            if freq <= remaining2:
                remaining2 -= freq
                removed2.add(num)
            else:
                remaining2 = 0
                break

        # Combine the remaining elements from both arrays into a set
        remaining_elements = set(nums1) | set(nums2)
        remaining_elements -= removed1
        remaining_elements -= removed2

        return len(remaining_elements)
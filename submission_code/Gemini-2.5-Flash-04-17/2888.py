from typing import List
import collections

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Find the dominant element and its total frequency in the whole array
        # The problem guarantees exactly one dominant element.
        counts = collections.Counter(nums)
        dom_elem = -1
        total_dom_freq = 0
        for elem, freq in counts.items():
            if freq * 2 > n:
                dom_elem = elem
                total_dom_freq = freq
                break # Found the unique dominant element

        # Step 2 & 3: Iterate through possible split indices and check validity
        # A split at index i means the left part is nums[0...i] and the right part is nums[i+1...n-1].
        # The valid range for i is 0 to n-2 (inclusive), as per the problem statement 0 <= i < n-1.
        left_dom_freq = 0 # Frequency of dom_elem in the current left part nums[0...i]

        for i in range(n - 1): # i is the index of the last element in the left part
            # When we consider the split after index i, nums[i] is in the left part.
            if nums[i] == dom_elem:
                left_dom_freq += 1

            # Calculate frequencies and lengths for the split at index i
            # Left part: nums[0...i], length is i + 1
            # Right part: nums[i+1...n-1], length is n - (i + 1)
            m_left = i + 1
            m_right = n - m_left # Length of the right part
            right_dom_freq = total_dom_freq - left_dom_freq # Frequency of dom_elem in the right part

            # Check dominance conditions for the split at index i
            # The split is valid if the dominant element (dom_elem) is dominant in *both* parts
            is_left_dominant = left_dom_freq * 2 > m_left
            is_right_dominant = right_dom_freq * 2 > m_right

            if is_left_dominant and is_right_dominant:
                return i # Found the minimum valid split index, return immediately

        # Step 4: If the loop finishes without finding a valid split
        return -1
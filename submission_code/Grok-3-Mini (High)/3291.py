import math
from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        arr = nums[:]  # Create a copy of the array to avoid modifying the input
        n = len(arr)
        i = 0
        while i < n:
            # Compute the popcount of the current element
            current_pop = bin(arr[i]).count('1')
            # Find the end of the segment with the same popcount
            j = i
            while j < n and bin(arr[j]).count('1') == current_pop:
                j += 1
            # Sort the subarray from index i to j-1 (slice arr[i:j])
            arr[i:j] = sorted(arr[i:j])
            # Move to the start of the next segment
            i = j
        # Check if the array is sorted
        for k in range(n - 1):
            if arr[k] > arr[k + 1]:
                return False
        return True
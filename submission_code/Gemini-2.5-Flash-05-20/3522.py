from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []

        # Iterate through all possible starting indices for subarrays of size k.
        # The loop runs from i = 0 up to n - k (inclusive).
        for i in range(n - k + 1):
            is_power_array = True # Assume the current subarray is a power array initially
            
            # Check for consecutiveness and ascending order.
            # We only need to check k-1 pairs of adjacent elements.
            # If k=1, range(k-1) becomes range(0), and this loop is skipped,
            # correctly leaving is_power_array as True for single-element arrays.
            for j in range(k - 1):
                # Access elements directly from the original nums array
                # The current element in the subarray is nums[i + j]
                # The next element in the subarray is nums[i + j + 1]
                if nums[i + j + 1] != nums[i + j] + 1:
                    is_power_array = False # Condition not met
                    break # No need to check further elements for this subarray
            
            # Determine the power based on the check
            if is_power_array:
                # If it's a consecutive and sorted array, its maximum element is its last element.
                # The last element of the current subarray (nums[i : i+k]) is nums[i + k - 1].
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        
        return results
from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        
        # A subarray of size 1 is always consecutive and sorted by definition.
        # Its power is the element itself. This is a simple and fast path.
        if k == 1:
            return nums

        # The core idea is to find "runs" of consecutive increasing numbers.
        # A subarray nums[i:i+k] is valid if the run ending at its last element
        # (nums[i+k-1]) is at least k long.

        # Step 1: Precompute the length of consecutive increasing runs.
        # runs[i] stores the length of the consecutive run ending at index i.
        # A run is a sequence like [x, x+1, x+2, ...].
        # For example, if nums = [1, 2, 3, 5, 6], then runs = [1, 2, 3, 1, 2].
        runs = [1] * n
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                runs[i] = runs[i - 1] + 1
        
        # Step 2: Iterate through all subarrays of size k and calculate their power.
        results = []
        for i in range(n - k + 1):
            # The current subarray is nums[i...i+k-1].
            # It ends at index i + k - 1.
            end_index = i + k - 1
            
            # The condition "consecutive and sorted in ascending order" for a
            # subarray of length k is met if and only if the run of consecutive
            # numbers ending at its last element is at least k long.
            if runs[end_index] >= k:
                # If the condition is met, the power is the maximum element.
                # In a sorted consecutive sequence, the last element is the maximum.
                power = nums[end_index]
                results.append(power)
            else:
                # If the condition is not met, the power is -1.
                results.append(-1)
                
        return results
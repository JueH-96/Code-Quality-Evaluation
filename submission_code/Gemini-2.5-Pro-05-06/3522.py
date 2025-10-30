from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        
        # Constraints: 1 <= n <= 500, 1 <= k <= n.
        # This means n-k+1 (number of subarrays) is always at least 1.

        results = [] 
        
        # Iterate through all possible start indices 'i' for subarrays of size k.
        # A subarray nums[i...i+k-1] starts at index i.
        # 'i' ranges from 0 to n-k.
        for i in range(n - k + 1):
            # Assume the current subarray is consecutive and sorted ascending.
            is_consecutive_and_sorted = True
            
            # If k=1, a single-element subarray is always valid.
            # The loop 'for j in range(k-1)' becomes 'for j in range(0)',
            # which doesn't execute. 'is_consecutive_and_sorted' remains True.
            
            # Check all k-1 adjacent pairs in the subarray nums[i...i+k-1].
            # The j-th pair (0-indexed) consists of subarray elements at
            # relative indices j and j+1.
            # These correspond to nums[i+j] and nums[i+j+1].
            for j in range(k - 1):
                current_element = nums[i + j]
                next_element = nums[i + j + 1]
                
                # If not consecutive and sorted (i.e., next != current + 1)
                if next_element != current_element + 1:
                    is_consecutive_and_sorted = False
                    break # Subarray is invalid, no need to check further pairs.
            
            if is_consecutive_and_sorted:
                # Valid subarray: power is its maximum element.
                # Since it's sorted ascending, max is the last element: nums[i+k-1].
                power = nums[i + k - 1]
                results.append(power)
            else:
                # Invalid subarray: power is -1.
                results.append(-1)
                
        return results
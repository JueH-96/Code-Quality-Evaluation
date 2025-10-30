from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize min_length to a value larger than any possible subarray length (max n)
        min_length = n + 1 

        # Iterate through all possible starting indices of a subarray
        for i in range(n):
            current_or = 0 # Initialize OR value for the subarray starting at i
            # Iterate through all possible ending indices for the current start index i
            for j in range(i, n):
                # Include the current element nums[j] in the bitwise OR
                current_or |= nums[j]

                # Check if the bitwise OR of the subarray nums[i...j] is at least k
                if current_or >= k:
                    # If it is, this subarray is "special". Calculate its length.
                    subarray_length = j - i + 1
                    # Update the minimum length found so far
                    min_length = min(min_length, subarray_length)
                    
                    # Optimization: Since we found a special subarray nums[i...j]
                    # starting at 'i', any longer subarray starting at 'i' (i.e.,
                    # ending at j' > j) will have an OR value that is
                    # (current_or | OR(nums[j+1...j'])) which is guaranteed to be
                    # >= current_or. Since current_or >= k, the OR value of longer
                    # subarrays starting at 'i' will also be >= k. However, these
                    # longer subarrays will have length j' - i + 1 > j - i + 1.
                    # Thus, nums[i...j] is the shortest special subarray starting
                    # at index 'i'. We don't need to check j' > j for this starting
                    # index i. We can break the inner loop and move to the next i.
                    break 

        # After checking all possible subarrays, if min_length is still the initial 
        # sentinel value, it means no special subarray was found.
        if min_length == n + 1:
            return -1
        else:
            # Otherwise, return the minimum length found
            return min_length
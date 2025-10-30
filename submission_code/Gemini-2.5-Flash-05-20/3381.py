from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_length = float('inf')  # Initialize with positive infinity to find the minimum length

        # Iterate over all possible starting indices of a subarray
        for start in range(n):
            current_or_sum = 0  # Initialize the bitwise OR sum for the current subarray
            
            # Iterate over all possible ending indices for the current starting index
            # This forms subarrays: [nums[start]], [nums[start], nums[start+1]], ..., [nums[start], ..., nums[n-1]]
            for end in range(start, n):
                # Include the current element nums[end] into the OR sum
                current_or_sum |= nums[end]
                
                # Check if the bitwise OR sum of the current subarray [start...end] is at least k
                if current_or_sum >= k:
                    # If it is, this is a special subarray. Calculate its length.
                    length = end - start + 1
                    # Update the minimum length found so far
                    min_length = min(min_length, length)
                    
                    # Optimization:
                    # Since the bitwise OR operation is non-decreasing (adding more elements
                    # can only increase or keep the OR sum the same),
                    # any further extension of this subarray (i.e., by increasing 'end')
                    # would only result in a longer subarray that also satisfies the condition.
                    # As we are looking for the *shortest* subarray, we can stop extending
                    # the current subarray from 'start' and move to the next 'start' index.
                    break
        
        # After checking all possible subarrays, if min_length is still float('inf'),
        # it means no special subarray was found that satisfies the condition.
        if min_length == float('inf'):
            return -1
        else:
            # Otherwise, return the minimum length found.
            # If min_length was updated, it would have been updated with an integer length,
            # so no explicit type conversion is needed here for the return value.
            return min_length
from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_len = float('inf')

        # Iterate through all possible starting indices of a subarray.
        for i in range(n):
            current_or = 0
            # Iterate through all possible ending indices for a subarray starting at i.
            for j in range(i, n):
                # The subarray is nums[i...j].
                # We update the bitwise OR by including the element nums[j].
                current_or |= nums[j]
                
                # Check if the current subarray is "special" (its OR sum is at least k).
                if current_or >= k:
                    # If it is, we have found a special subarray of length j - i + 1.
                    length = j - i + 1
                    min_len = min(min_len, length)
                    
                    # Optimization: For a fixed start 'i', we have found the shortest
                    # special subarray. Any further extension (increasing 'j') would
                    # only result in a longer one. So, we can break and check the
                    # next starting position 'i+1'.
                    break
        
        # If min_len was never updated from its initial value of infinity,
        # it means no special subarray was found.
        if min_len == float('inf'):
            return -1
        else:
            return min_len
from typing import List

class Solution:
    """
    Calculates the number of incremovable subarrays.
    An incremovable subarray is one such that after removing it, the remaining elements form a strictly increasing sequence.
    """
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        """
        Args:
            nums: A list of positive integers.

        Returns:
            The total number of incremovable subarrays.
        """
        n = len(nums)
        
        # Find the length `p` of the longest strictly increasing prefix `nums[0...p-1]`.
        # `p` is the first index from the left where the strictly increasing property breaks.
        # If the entire array is strictly increasing, `p` will be `n`.
        # For example, in [1, 2, 4, 3, 5], p = 3 because nums[3]=3 is not > nums[2]=4. The prefix is [1, 2, 4].
        # In [1, 2, 3], p=3.
        p = n 
        for k in range(1, n):
            if nums[k] <= nums[k-1]:
                p = k
                break
        
        # If the entire array is strictly increasing (p == n), then removing any subarray
        # will result in a sequence that is also strictly increasing.
        # An array with N elements has N*(N+1)/2 non-empty subarrays.
        if p == n:
            return n * (n + 1) // 2

        # Find the starting index `s` of the longest strictly increasing suffix `nums[s...n-1]`.
        # `s` is the smallest index such that the suffix starting at `s` is strictly increasing.
        # For example, in [1, 2, 4, 3, 5], s=3 because nums[3...4] = [3, 5] is strictly increasing, but nums[2...4]=[4, 3, 5] is not.
        # In [1, 2, 3], s=0.
        # If the entire array is strictly increasing, s=0.
        s = 0
        # Iterate from the second to last element down to the first element.
        for k in range(n - 2, -1, -1):
             # If we find an element nums[k] that is not strictly smaller than the next element nums[k+1]
             if nums[k] >= nums[k+1]:
                 # The strictly increasing suffix must start after index k.
                 s = k + 1
                 break

        # Initialize count of incremovable subarrays.
        count = 0
        
        # Iterate through all possible start indices `i` of the subarray to remove (nums[i...j]).
        # The prefix remaining after removal is `nums[0...i-1]`. This prefix must be strictly increasing.
        # This condition holds if `i <= p` (length of the prefix is `i`, indices are `0` to `i-1`).
        # Since `nums[0...p-1]` is the longest strictly increasing prefix, any prefix `nums[0...i-1]` with `i <= p` will also be strictly increasing.
        # So we only need to consider `i` from 0 up to `p`.
        for i in range(p + 1):
            # Iterate through all possible end indices `j` of the subarray to remove.
            # The suffix remaining after removal is `nums[j+1...n-1]`. This suffix must be strictly increasing.
            # This condition holds if `j+1 >= s`, which means `j >= s-1`.
            # Also, for a valid subarray nums[i...j], we must have `j >= i`.
            # Therefore, the starting value for j in the inner loop is max(i, s-1).
            start_j = max(i, s - 1)
            
            for j in range(start_j, n):
                # The subarray removed is nums[i...j].
                # The remaining parts are Prefix: nums[0...i-1] and Suffix: nums[j+1...n-1].
                
                # The combined sequence (Prefix + Suffix) needs to be strictly increasing.
                # We already know:
                # 1. Prefix `nums[0...i-1]` is strictly increasing (because i <= p).
                # 2. Suffix `nums[j+1...n-1]` is strictly increasing (because j >= s-1 => j+1 >= s).
                # We only need to check the connection point if both Prefix and Suffix are non-empty.
                
                # Check if the removal of nums[i...j] results in a strictly increasing sequence.
                is_incremovable = False
                if i == 0:
                    # Case 1: Prefix is empty (i=0). The remaining sequence is just the Suffix nums[j+1...n-1].
                    # This suffix is guaranteed strictly increasing because j >= s-1 implies j+1 >= s.
                    is_incremovable = True
                else: # i > 0, Prefix is non-empty (nums[0...i-1] exists).
                    if j == n - 1:
                        # Case 2: Suffix is empty (j=n-1). The remaining sequence is just the Prefix nums[0...i-1].
                        # This prefix is guaranteed strictly increasing because i <= p.
                        is_incremovable = True
                    else: # j < n-1, Suffix is non-empty (nums[j+1...n-1] exists).
                        # Case 3: Both Prefix and Suffix are non-empty.
                        # The overall sequence is strictly increasing if and only if
                        # the last element of the Prefix is strictly less than the first element of the Suffix.
                        # Check the connection: nums[i-1] < nums[j+1]
                        if nums[i-1] < nums[j+1]:
                            is_incremovable = True
                            
                if is_incremovable:
                    count += 1
                    
        return count
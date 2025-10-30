from typing import List

class Solution:
    def _is_strictly_increasing(self, nums_list: List[int], start_idx: int, sub_len: int) -> bool:
        """
        Helper function to check if the subarray nums_list[start_idx : start_idx + sub_len]
        is strictly increasing.
        """
        # A subarray of length 1 (or 0, though k>=1 by problem constraints)
        # is considered strictly increasing.
        if sub_len <= 1:
            return True
        
        # Iterate through adjacent pairs in the subarray
        # We need to check (sub_len - 1) pairs.
        # Example: for sub_len=3, pairs are (nums[0],nums[1]) and (nums[1],nums[2])
        # Loop goes for i = 0 to sub_len-2.
        for i in range(sub_len - 1):
            # Current element is at nums_list[start_idx + i]
            # Next element is at nums_list[start_idx + i + 1]
            idx1 = start_idx + i
            idx2 = start_idx + i + 1
            if nums_list[idx1] >= nums_list[idx2]:
                return False  # Not strictly increasing
        return True  # All pairs checked, it's strictly increasing

    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Problem constraints:
        # 1 < 2 * k <= nums.length
        # This implies k >= 1 (since k is an integer, k > 1/2 means k >= 1).
        # This also implies n >= 2*k, so there's enough space for two adjacent subarrays.

        # The loop for 'a' (start index of the first subarray).
        # The first subarray is nums[a : a+k].
        # The second subarray is nums[a+k : a+2k].
        # The last element of the second subarray is at index a+2k-1.
        # So, a+2k-1 < n => a+2k <= n => a <= n - 2k.
        # 'a' ranges from 0 to n - 2k, inclusive.
        # The number of iterations is (n - 2*k - 0) + 1 = n - 2*k + 1.
        # This count is always at least 1 because n >= 2*k from constraints.
        
        for a in range(n - 2*k + 1):
            # Check the first subarray: nums[a ... a+k-1]
            # Its starting index in `nums` is `a`, and its length is `k`.
            is_sub1_increasing = self._is_strictly_increasing(nums, a, k)
            
            if is_sub1_increasing:
                # If the first subarray is strictly increasing, check the second one.
                # The second subarray: nums[a+k ... a+k+k-1]
                # Its starting index in `nums` is `a+k`, and its length is `k`.
                is_sub2_increasing = self._is_strictly_increasing(nums, a + k, k)
                
                if is_sub2_increasing:
                    # Found a pair of adjacent, strictly increasing subarrays
                    return True
        
        # No such pair found after checking all possible starting positions 'a'
        return False
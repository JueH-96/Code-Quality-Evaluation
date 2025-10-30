from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the index `l` such that `nums[0...l]` is the longest strictly increasing prefix.
        # `l` is the last index of this prefix.
        # The prefix `nums[:i]` (indices 0 to i-1) is strictly increasing if `i-1 <= l`, i.e., `i <= l+1`.
        # So, valid start indices `i` for the subarray to remove range from 0 up to `l+1`.
        l = -1 # Represents the end index of the prefix nums[:i] being empty
        while l + 1 < n and (l == -1 or nums[l] < nums[l+1]):
             l += 1
        # After the loop, `nums[0...l]` is strictly increasing.
        # `l` is the index of the last element in this maximal strict prefix.
        # The possible values for `i` (exclusive end of prefix) are 0, 1, ..., l+1.

        # Find the index `r` such that `nums[r...n-1]` is the longest strictly increasing suffix.
        # `r` is the first index of this suffix.
        # The suffix `nums[j:]` (indices j to n-1) is strictly increasing if `j >= r`.
        # So, valid start indices `j` for the suffix range from `r` up to `n`.
        r = n # Represents the start index of the suffix nums[j:] being empty
        while r - 1 >= 0 and (r == n or nums[r-1] < nums[r]):
            r -= 1
        # After the loop, `nums[r...n-1]` is strictly increasing.
        # `r` is the index of the first element in this maximal strict suffix.
        # The possible values for `j` (start index of suffix) are r, r+1, ..., n.
        
        count = 0
        
        # Iterate through all possible start indices `i` of the subarray to remove.
        # The prefix kept is `nums[:i]`. It is strictly increasing if `i <= l + 1`.
        # `i` ranges from 0 up to `l + 1`. `range(l + 2)` generates `0, 1, ..., l+1`.
        for i in range(l + 2):
            
            # For a fixed `i`, the prefix `nums[:i]` is guaranteed strictly increasing.
            # We need to find the number of valid end indices `j` (exclusive)
            # for the subarray `nums[i:j]`.
            
            # The subarray `nums[i:j]` must be non-empty, so `i < j`, i.e., `j >= i + 1`.
            # The suffix kept is `nums[j:]`. It must be strictly increasing, which requires `j >= r`.
            
            # So, we need `j` such that `j >= i + 1` and `j >= r`.
            # This implies `j >= max(i + 1, r)`.
            # The index `j` is the start of the suffix, and can go up to `n`.
            
            # The smallest possible value for `j` is `max(i + 1, r)`.
            min_j = max(i + 1, r)

            # Iterate through potential end indices `j` of the subarray.
            # `j` ranges from `min_j` up to `n`.
            # `range(min_j, n + 1)` generates `min_j, min_j + 1, ..., n`.
            for j in range(min_j, n + 1):
                # For this pair (i, j):
                # - The prefix `nums[:i]` is strictly increasing (because `i <= l + 1`).
                # - The suffix `nums[j:]` is strictly increasing (because `j >= r`).
                # - The removed subarray `nums[i:j]` is non-empty (because `j >= i + 1 > i`).
                
                # The remaining sequence `nums[:i] + nums[j:]` is strictly increasing
                # if and only if the connection between the prefix and suffix holds,
                # provided both are non-empty.
                # The connection condition is `nums[i-1] < nums[j]`.
                # This check is only needed if `i > 0` (prefix is non-empty)
                # AND `j < n` (suffix is non-empty).
                
                is_valid_connection = True
                if i > 0 and j < n:
                    if nums[i - 1] >= nums[j]:
                        is_valid_connection = False
                
                # If the connection is valid (or not applicable), the subarray `nums[i:j]` is incremovable.
                if is_valid_connection:
                    count += 1
        
        return count
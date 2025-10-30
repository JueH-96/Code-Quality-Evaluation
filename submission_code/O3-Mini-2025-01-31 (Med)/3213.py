from typing import List
import bisect

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Find the overall maximum element in nums.
        max_val = max(nums)
        # Build a prefix count array for occurrences of max_val.
        # prefix[i] = number of times max_val appears in nums[:i]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if nums[i] == max_val else 0)
        
        count = 0
        # For every possible starting index L, find smallest R such that 
        # the subarray nums[L:R+1] contains at least k occurrences of max_val.
        # That is, we need: prefix[R+1] - prefix[L] >= k.
        # Since prefix is non-decreasing, we can binary search for needed R.
        for L in range(n):
            # current count before L is prefix[L], so we need prefix[R+1] >= prefix[L] + k.
            target = prefix[L] + k
            # Binary search in prefix array from index L+1 to n to find the smallest index r such that prefix[r] >= target.
            # The binary search is done on the entire prefix list.
            R_index = bisect.bisect_left(prefix, target, L+1, n+1)
            if R_index <= n:
                # If found, then all subarrays starting at L with R >= R_index-1 are valid.
                # Note: R_index is the index in prefix, and it corresponds to subarray ending at index (R_index - 1).
                count += (n - (R_index - 1))
        return count
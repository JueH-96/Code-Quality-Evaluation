from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n  # No removals possible if 0 or 1 element
        
        # Find the maximum length of consecutive equal elements
        max_count = 1
        current_count = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
            else:
                current_count = 1
        
        M = max_count  # Maximum frequency of any element
        if 2 * M <= n:
            return n % 2  # If 2*M <= n, min length is the parity of n
        else:
            return 2 * M - n  # Otherwise, min length is 2*M - n
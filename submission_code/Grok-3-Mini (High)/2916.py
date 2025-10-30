from typing import List
import functools

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 2:
            return True
        
        # Compute prefix sums for O(1) range sum queries
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # Memoized recursive function to check if subarray from start to end can be split
        @functools.lru_cache(None)
        def check(start, end):
            length = end - start + 1
            if length <= 2:
                return True
            for k in range(start, end):  # k is the end index of the left part
                sum_left = prefix[k + 1] - prefix[start]
                sum_right = prefix[end + 1] - prefix[k + 1]
                len_left = k - start + 1
                len_right = end - k
                if (len_left == 1 or sum_left >= m) and (len_right == 1 or sum_right >= m) and check(start, k) and check(k + 1, end):
                    return True  # Valid split found
            return False  # No valid split found
        
        return check(0, n - 1)
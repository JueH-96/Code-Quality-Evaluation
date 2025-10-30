from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1
        
        # Find the global dominant element via Boyer-Moore majority vote
        # (since there's guaranteed exactly one dominant element).
        candidate = None
        count = 0
        for x in nums:
            if count == 0:
                candidate = x
                count = 1
            elif candidate == x:
                count += 1
            else:
                count -= 1
        
        # Confirm it's truly dominant and get its total frequency.
        total = 0
        for x in nums:
            if x == candidate:
                total += 1
        
        # (By problem guarantee, 2 * total > n must hold.)
        
        # Now sweep prefixes to find the earliest valid split.
        prefix_count = 0
        for i, x in enumerate(nums):
            if x == candidate:
                prefix_count += 1
            
            # we can only split at i if i < n-1
            if i < n - 1:
                left_len = i + 1
                right_len = n - left_len
                left_dom = (prefix_count * 2 > left_len)
                right_dom = ((total - prefix_count) * 2 > right_len)
                if left_dom and right_dom:
                    return i
        
        return -1
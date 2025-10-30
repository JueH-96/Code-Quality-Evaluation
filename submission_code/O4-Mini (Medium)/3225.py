from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        Return the length of the longest subarray in which every element appears
        at most k times.
        We use a sliding window [left, right), a frequency map, and a counter
        to track how many elements exceed frequency k in the current window.
        """
        freq = {}
        left = 0
        bad = 0   # number of values whose count > k
        ans = 0
        
        for right, x in enumerate(nums):
            # include nums[right]
            cnt = freq.get(x, 0) + 1
            freq[x] = cnt
            if cnt == k + 1:
                bad += 1
            
            # while window invalid, shrink from left
            while bad > 0:
                y = nums[left]
                cnty = freq[y]
                if cnty == k + 1:
                    bad -= 1
                freq[y] = cnty - 1
                left += 1
            
            # now window [left..right] is valid
            ans = max(ans, right - left + 1)
        
        return ans
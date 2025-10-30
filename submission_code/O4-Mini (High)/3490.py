from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count how many evens and how many odds
        cnt_even = sum(1 for x in nums if x % 2 == 0)
        cnt_odd = len(nums) - cnt_even
        
        # Case c = 0: take all evens or all odds -> max(cnt_even, cnt_odd)
        # Case c = 1: take an alternating subsequence.
        #   If counts are equal, you can alternate fully: length = 2 * cnt_even
        #   Otherwise you get 2 * min(cnt_even, cnt_odd) + 1
        if cnt_even == cnt_odd:
            alt_len = 2 * cnt_even
        else:
            alt_len = 2 * min(cnt_even, cnt_odd) + 1
        
        return max(cnt_even, cnt_odd, alt_len)
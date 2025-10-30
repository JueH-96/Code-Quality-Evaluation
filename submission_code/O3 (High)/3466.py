from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # prev_and_count will store:  AND value  ->  how many subarrays
        # whose AND equals that value and END at the previous index
        prev_and_count = {}
        ans = 0
        
        for num in nums:
            curr_and_count = {num: 1}        # the subarray consisting of [num] itself
            
            # extend every sub-array that ended at the previous position with `num`
            for val, cnt in prev_and_count.items():
                new_val = val & num
                curr_and_count[new_val] = curr_and_count.get(new_val, 0) + cnt
            
            # all sub-arrays that end at the current index and have AND == k
            ans += curr_and_count.get(k, 0)
            
            # move on to the next position
            prev_and_count = curr_and_count
        
        return ans
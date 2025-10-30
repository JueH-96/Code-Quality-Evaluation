from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        # prev maps a bitwise-AND value to how many subarrays ending at the previous index have that AND
        prev = {}
        
        for num in nums:
            # curr will map AND-values for subarrays ending at the current index
            curr = { num: 1 }
            # extend all previous subarrays by including the current number
            for val, cnt in prev.items():
                new_val = val & num
                curr[new_val] = curr.get(new_val, 0) + cnt
            
            # add counts of those ending here that equal k
            ans += curr.get(k, 0)
            prev = curr
        
        return ans
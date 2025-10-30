from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        count = 0
        run_len = 1  # length of current alternating run
        
        for i in range(1, n):
            if nums[i] != nums[i - 1]:
                # continue the alternating run
                run_len += 1
            else:
                # close off the current run and start a new one
                count += run_len * (run_len + 1) // 2
                run_len = 1
        
        # add the last run
        count += run_len * (run_len + 1) // 2
        
        return count
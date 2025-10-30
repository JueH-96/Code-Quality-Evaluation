from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        # freq[x] = number of times x appears in nums, for x in [0..n]
        freq = [0] * (n + 1)
        for x in nums:
            freq[x] += 1
        
        ways = 0
        c_lt = 0  # count of numbers < k for the current k
        # try all possible k = number of selected students from 0 to n
        for k in range(n + 1):
            # condition: exactly k numbers must be < k, and no number equals k
            if c_lt == k and freq[k] == 0:
                ways += 1
            # advance c_lt to include freq[k] for the next k+1
            c_lt += freq[k]
        
        return ways
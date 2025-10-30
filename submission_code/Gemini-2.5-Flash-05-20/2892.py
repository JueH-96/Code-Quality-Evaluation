import collections
from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Step 1: Determine the candidate n.
        # According to the definition, if nums is a permutation of base[n],
        # then n must be the maximum value present in nums.
        # Constraints guarantee nums.length >= 1, so max(nums) is safe.
        max_val = max(nums)

        # Step 2: Check length compatibility.
        # The base[n] array has a length of n + 1.
        # So, the given nums array must have a length equal to max_val + 1.
        if len(nums) != max_val + 1:
            return False

        # Step 3: Count frequencies of elements in nums.
        # collections.Counter is an efficient way to do this.
        counts = collections.Counter(nums)

        # Step 4: Verify counts based on the definition of base[n].
        # base[n] contains:
        # - Numbers from 1 to n-1 exactly once.
        # - The number n exactly twice.

        # Check numbers from 1 to max_val - 1.
        # The loop range (1, max_val) correctly handles cases where max_val is 1
        # (range becomes empty, so this check implicitly passes for n=1).
        for i in range(1, max_val):
            # If any number from 1 to max_val-1 is not present or appears more than once,
            # then nums is not good.
            if counts[i] != 1:
                return False
        
        # Check the number max_val itself. It must appear exactly twice.
        if counts[max_val] != 2:
            return False
            
        # If all checks pass, then nums is a good array.
        return True
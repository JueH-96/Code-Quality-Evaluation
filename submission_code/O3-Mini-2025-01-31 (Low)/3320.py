from typing import List
from collections import Counter

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # The problem can be re-interpreted as:
        # "Pair up the numbers (each number used at most once) so that all pairs have the same sum.
        # Return the maximum number of pairs possible under that restriction."
        #
        # To solve this, we iterate over all possible target sums 
        # (the sum of any valid pair) and try to greedily form pairs using a Counter.
        #
        # The maximum target sum possible is max(nums) + max(nums)
        # and the minimum target sum is min(nums) + min(nums).
        # Given the constraints (nums length up to 100 and values up to 1000), 
        # it is efficient to try all possible target sums.
        
        freq_original = Counter(nums)
        max_ops = 0
        
        # Possible range of sums:
        # lowest possible sum:
        min_sum = min(nums) * 2
        # highest possible sum:
        max_sum = max(nums) * 2
        
        # try every possible target sum in the inclusive range.
        for target in range(min_sum, max_sum + 1):
            # create a copy of frequency counter to simulate pairing without affecting original.
            freq = freq_original.copy()
            ops = 0  # count of pairs for this target
            # iterate over numbers in sorted order to guarantee that each pair is only counted once.
            for num in sorted(freq):
                if freq[num] == 0:
                    continue
                complement = target - num
                # If complement is less than num, then it would have been handled in previous iterations.
                if complement < num:
                    continue
                if complement not in freq:
                    continue
                if num == complement:
                    # In case both numbers are the same, we can pair freq[num] // 2 pairs.
                    pairs = freq[num] // 2
                    ops += pairs
                    freq[num] -= pairs * 2
                else:
                    # For distinct numbers, form pairs: min(freq[num], freq[complement])
                    pairs = min(freq[num], freq[complement])
                    ops += pairs
                    freq[num] -= pairs
                    freq[complement] -= pairs
            max_ops = max(max_ops, ops)
        return max_ops
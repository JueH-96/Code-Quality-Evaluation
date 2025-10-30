from typing import List
from collections import Counter

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # Explanation:
        # We want to choose a target integer x and try to maximize:
        #    freq(x) + min(numOperations, count_in_range(x) - freq(x))
        # Here:
        #   freq(x) = number of elements originally equal to x (free, no op needed).
        #   count_in_range(x) = number of elements in nums that satisfy |nums[i]-x| <= k,
        #        i.e. they can be turned into x using one operation.
        # With at most "numOperations" operations (each on a distinct index) available,
        # we can convert at most numOperations of those that aren’t originally x.
        # So for a fixed x, the max frequency we can achieve is:
        #     freq(x) + min(numOperations, count_in_range(x) - freq(x))
        #
        # Our plan:
        #   (1) Sort nums.
        #   (2) Precompute a frequency dictionary of the original values.
        #   (3) Note that an index i qualifies for conversion to x if and only if nums[i] is in [x-k, x+k].
        #       Given sorted nums, as x varies, the count of such indices changes in a monotonic way.
        #   (4) We consider candidate x in the range [min(nums)-k, max(nums)+k].
        #       (Even if x does not appear in nums, it might yield a better chance if many nums fall within [x-k, x+k].)
        #   (5) Use a two-pointer sliding-window method to count, for each candidate x, the number
        #       of nums that lie in [x-k, x+k].
        
        nums.sort()
        n = len(nums)
        freq = Counter(nums)
        
        # Candidate x can reasonably be in the interval from (min(nums)-k) to (max(nums)+k)
        candidate_low = nums[0] - k
        candidate_high = nums[-1] + k
        
        res = 0
        left = 0
        right = 0   # both pointers for the sliding window over sorted nums
        
        # Iterate through every candidate x in the chosen range.
        for x in range(candidate_low, candidate_high + 1):
            # Adjust the left pointer: 
            # We want the first index such that nums[left] is at least x-k.
            while left < n and nums[left] < x - k:
                left += 1
            # Adjust the right pointer:
            # We want right to be the first index where nums[right] is greater than x+k.
            while right < n and nums[right] <= x + k:
                right += 1
                
            # current_window is count of elements in sorted_nums that are in [x-k, x+k]
            current_window = right - left
            # Number of free elements already equal to x.
            already_x = freq.get(x, 0)
            # To convert, we can use at most numOperations operations on other eligible numbers.
            candidate_frequency = already_x + min(numOperations, current_window - already_x)
            res = max(res, candidate_frequency)
            
            # Early exit: the maximum frequency cannot exceed n.
            if res == n:
                return res
            
        return res
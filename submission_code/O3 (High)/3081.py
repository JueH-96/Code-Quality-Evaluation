from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Pointers for the two halves of the array
        left  = 0          # candidates for the smaller element
        right = n // 2     # candidates for the larger element
        pairs = 0          # number of successful pairs found
        
        # Try to match each element in the first half with
        # the earliest possible strictly larger element in the second half.
        while left < n // 2 and right < n:
            if nums[left] < nums[right]:
                pairs += 1     # a pair is formed
                left  += 1
                right += 1
            else:
                right += 1     # need a larger element for current left
        
        # Each pair removes two elements.
        return n - 2 * pairs
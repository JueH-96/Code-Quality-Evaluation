from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # The problem allows us to delete any elements so that the remaining array’s 
        # contiguous subarray (which will be our chosen subsequence from the original order)
        # contains only unique elements and has maximum sum.
        #
        # We can view this as picking any nonempty subsequence (preserving order)
        # such that each value is used at most once.
        #
        # Notice that:
        # - For any positive number, including it increases the sum.
        # - Including 0 does not change the sum.
        # - Including a negative number lowers the sum.
        #
        # Therefore, if there is at least one positive number,
        # the best is to simply take one occurrence of each positive number
        # (order can be arranged by deleting other elements so that they are contiguous).
        #
        # In the case there are no positive numbers,
        # if 0 is present the best is [0] (sum = 0), 
        # and otherwise (all negatives) the best is to take the single element that is the maximum (least negative).
        
        positive_set = {x for x in nums if x > 0}
        if positive_set:
            return sum(positive_set)
        elif 0 in nums:
            return 0
        else:
            return max(nums)
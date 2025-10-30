import collections
from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        Solves the problem by using dynamic programming on the sorted input array.
        
        The state `dp[x]` represents the length of the longest consecutive sequence
        that can be formed ending with the number `x`. We iterate through the sorted
        `nums` array to build up these sequences.
        """
        
        # Sort the array to process numbers in an order that's conducive to
        # building consecutive sequences.
        nums.sort()
        
        # dp[x] will store the length of the longest consecutive sequence
        # that can be formed ending at value x. A defaultdict(int) initializes
        # non-existent keys with a value of 0.
        dp = collections.defaultdict(int)
        
        for n in nums:
            # For each number `n`, we can use it as `n` or `n+1`.
            
            # To form a sequence ending at `n+1` using the current `n`, we extend a
            # sequence that ended at `n`. The new length is `dp[n] + 1`.
            # We must perform this update first to use the value of `dp[n]` from
            # before it's updated by the current element `n`.
            dp[n + 1] = dp[n] + 1
            
            # To form a sequence ending at `n` using the current `n`, we extend a
            # sequence that ended at `n-1`. The new length is `dp[n-1] + 1`.
            dp[n] = dp[n - 1] + 1
            
        # The result is the maximum length of any consecutive sequence we could form.
        # This is the maximum value in our dp table.
        # Since nums is non-empty (per constraints), dp will not be empty.
        return max(dp.values())
from collections import defaultdict
from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Sort the array to process elements in increasing order.
        # This helps in building consecutive sequences.
        nums.sort()

        # dp[val] will store the maximum length of a consecutive sequence
        # ending with 'val'.
        # We use defaultdict(int) so that accessing a non-existent key
        # returns 0, which is useful for base cases like dp[x-1] when x-1 is not yet in dp.
        dp = defaultdict(int)

        # Initialize the maximum length found so far.
        ans = 0

        # Iterate through each number in the sorted array.
        for x in nums:
            # Option 1: Use the current 'x' as 'x' (no increment).
            # This 'x' would extend a consecutive sequence ending at 'x-1'.
            # The length of this potential sequence would be dp[x-1] + 1.

            # Option 2: Use the current 'x' as 'x+1' (increment by 1).
            # This 'x+1' would extend a consecutive sequence ending at 'x'.
            # The length of this potential sequence would be dp[x] + 1.
            # It's crucial that the dp[x] used here is the value that was computed
            # *before* the current 'x' element potentially updates dp[x] itself.
            # So, we save dp[x] before updating dp[x].
            
            val_x_before_update = dp[x]

            # Update dp[x]:
            # The maximum length of a sequence ending at 'x' can be either:
            #   a) Its previous maximum length (from other occurrences of 'x' or other numbers).
            #   b) A new sequence formed by extending a sequence ending at 'x-1' with the current 'x'.
            dp[x] = max(dp[x], dp[x-1] + 1)

            # Update dp[x+1]:
            # The maximum length of a sequence ending at 'x+1' can be either:
            #   a) Its previous maximum length.
            #   b) A new sequence formed by extending a sequence ending at 'x' (using elements processed
            #      before the current 'x' element) with the current 'x' (transformed to 'x+1').
            dp[x+1] = max(dp[x+1], val_x_before_update + 1)

            # Update the overall maximum answer by considering the newly updated
            # lengths for dp[x] and dp[x+1].
            ans = max(ans, dp[x], dp[x+1])
        
        return ans
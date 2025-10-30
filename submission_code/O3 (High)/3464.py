from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        dp_pos :  maximum total cost for the prefix that ends at the current
                  position with the current element taken with a PLUS sign
                  (i.e. it is at an even index inside the current sub-array)

        dp_neg :  maximum total cost for the prefix that ends at the current
                  position with the current element taken with a MINUS sign
                  (i.e. it is at an odd index inside the current sub-array)
        """
        
        # Only one element – it has to be taken with a ‘+’ sign
        dp_pos = nums[0]
        dp_neg = float('-inf')          # impossible after the first element
        
        for x in nums[1:]:
            best_prefix = max(dp_pos, dp_neg)      # prefix finished before x
            
            # x is taken with a ‘+’ sign
            new_pos = best_prefix + x              # either start new sub-array
                                                   # or continue the old one
            
            # x is taken with a ‘–’ sign
            new_neg = dp_pos - x                   # must continue the old sub-array
            
            dp_pos, dp_neg = new_pos, new_neg
        
        return max(dp_pos, dp_neg)
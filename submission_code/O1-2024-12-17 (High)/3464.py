class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        We use dynamic programming with two running states:
          - plus[i]:  the maximum total cost up to index i if the subarray
                      ending at i places a '+' sign on nums[i].
          - minus[i]: the maximum total cost up to index i if the subarray
                      ending at i places a '-' sign on nums[i].
        
        Since each subarray must alternate signs starting with '+', if
        we end at i with a '+' sign, we could either:
           (a) continue from i-1 which had a '-' sign, or
           (b) start a new subarray at i (and take the best total cost
               up to i-1 plus +nums[i]).

        Similarly, ending at i with '-' sign forces the previous element
        in that subarray (i-1) to have had a '+' sign.

        We also track best[i] = max(plus[i], minus[i]) as the maximum total
        cost covering exactly up to i (no gaps allowed). The final answer
        is best[n-1].
        """

        import math
        n = len(nums)
        if n == 1:
            # Only one element, so cost is simply that single element (with a '+' sign).
            return nums[0]
        
        # Initialize DP for i = 0
        plus = nums[0]          # subarray [0..0] => +nums[0]
        minus = float('-inf')   # cannot end with '-' on a single-element subarray
        best_ = plus            # best total cost up to index 0
        
        for i in range(1, n):
            # Compute the new minus and plus if we include nums[i]
            new_minus = plus - nums[i]
            new_plus = max(minus + nums[i], best_ + nums[i])
            
            # Update the best partition cost up to i
            new_best = max(new_minus, new_plus)
            
            # Advance
            minus, plus, best_ = new_minus, new_plus, new_best
        
        return best_
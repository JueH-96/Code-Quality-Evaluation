class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # We will keep track of two running "best" scores:
        #  - best_even: the max score achievable ending with an even-valued element
        #  - best_odd : the max score achievable ending with an odd-valued element
        
        # Initialize based on the parity of the first element
        if nums[0] % 2 == 0:
            best_even = nums[0]
            best_odd  = float('-inf')
        else:
            best_even = float('-inf')
            best_odd  = nums[0]
        
        # Iterate over the array from the 2nd element onward
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:
                # If current value is even:
                # either continue from best_even with no penalty
                #   or continue from best_odd with a penalty of x
                current = max(best_even + nums[i], best_odd + nums[i] - x)
                best_even = max(best_even, current)
            else:
                # If current value is odd:
                # either continue from best_odd with no penalty
                #   or continue from best_even with a penalty of x
                current = max(best_odd + nums[i], best_even + nums[i] - x)
                best_odd = max(best_odd, current)
        
        # The answer is the max of best_even and best_odd
        return max(best_even, best_odd)
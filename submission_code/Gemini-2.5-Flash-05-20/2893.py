import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)

        # dp_even_score: maximum score ending with an even number
        # dp_odd_score: maximum score ending with an odd number
        # Initialize with -infinity, as these paths are not yet established
        # (other than the initial nums[0]).
        dp_even_score = -math.inf
        dp_odd_score = -math.inf

        # The first number nums[0] is always visited.
        # Its score forms the base for our DP states.
        first_num = nums[0]
        if first_num % 2 == 0:
            dp_even_score = first_num
        else:
            dp_odd_score = first_num
        
        # overall_max_score tracks the absolute maximum score found so far.
        # This is because we can choose to stop visiting elements at any point.
        overall_max_score = first_num

        # Iterate from the second number in the array
        for i in range(1, n):
            current_val = nums[i]
            current_parity = current_val % 2

            # Store the current dp values before potentially updating them.
            # This ensures that calculations for the current step use values from the previous step.
            prev_dp_even = dp_even_score
            prev_dp_odd = dp_odd_score

            if current_parity == 0:  # current_val is even
                # Calculate potential score if we end with current_val (even):
                # Option 1: Extend a path that ended with an even number (no penalty)
                score_from_even = -math.inf
                if prev_dp_even != -math.inf:
                    score_from_even = prev_dp_even + current_val
                
                # Option 2: Extend a path that ended with an odd number (incurs penalty x)
                score_from_odd = -math.inf
                if prev_dp_odd != -math.inf:
                    score_from_odd = prev_dp_odd + current_val - x
                
                # Update dp_even_score with the maximum of these two options.
                # If both options are -inf, dp_even_score will also be -inf.
                dp_even_score = max(score_from_even, score_from_odd)

                # dp_odd_score remains unchanged because current_val is even,
                # so it cannot be the end of an odd-parity path.
                # (Implicitly, dp_odd_score retains its value from prev_dp_odd)

            else:  # current_parity == 1, current_val is odd
                # Calculate potential score if we end with current_val (odd):
                # Option 1: Extend a path that ended with an odd number (no penalty)
                score_from_odd = -math.inf
                if prev_dp_odd != -math.inf:
                    score_from_odd = prev_dp_odd + current_val

                # Option 2: Extend a path that ended with an even number (incurs penalty x)
                score_from_even = -math.inf
                if prev_dp_even != -math.inf:
                    score_from_even = prev_dp_even + current_val - x
                
                # Update dp_odd_score with the maximum of these two options.
                dp_odd_score = max(score_from_odd, score_from_even)

                # dp_even_score remains unchanged.
                # (Implicitly, dp_even_score retains its value from prev_dp_even)
            
            # Update the overall maximum score encountered so far.
            # We can choose to stop at current_val, so we consider both dp states.
            if dp_even_score != -math.inf:
                overall_max_score = max(overall_max_score, dp_even_score)
            if dp_odd_score != -math.inf:
                overall_max_score = max(overall_max_score, dp_odd_score)
        
        return overall_max_score
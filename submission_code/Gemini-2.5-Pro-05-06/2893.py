import math

class Solution:
  def maxScore(self, nums: list[int], x: int) -> int:
    n = len(nums)
    
    # dp_even stores the maximum score of a path ending with an even number.
    # dp_odd stores the maximum score of a path ending with an odd number.
    # A path must start with nums[0].
    
    # Initialize based on nums[0]
    neg_inf = -math.inf

    # We use floats for dp_even/dp_odd because -math.inf is a float.
    # All inputs are integers, so intermediate sums/differences will maintain precision
    # as long as they are within the range where floats can exactly represent integers
    # (approx up to 2^53 or 9*10^15, which is larger than max possible score of 10^11).
    if nums[0] % 2 == 0:  # nums[0] is even
        dp_even = float(nums[0])
        dp_odd = neg_inf
    else:  # nums[0] is odd
        dp_odd = float(nums[0])
        dp_even = neg_inf

    # The overall maximum score found. Initialized with the score from just visiting nums[0].
    # nums[0] is positive (>=1), so max_total_score starts positive.
    max_total_score = float(nums[0])

    # Iterate from the second element
    for i in range(1, n):
        current_num = nums[i]
        
        current_score_at_i = neg_inf # Max score for a path ending at nums[i]

        if current_num % 2 == 0:  # current_num is even
            # Option 1: Previous element in path was even.
            # Score = dp_even + current_num (No penalty: even -> even)
            score_if_prev_even = dp_even + current_num
            
            # Option 2: Previous element in path was odd.
            # Score = dp_odd - x + current_num (Penalty x: odd -> even)
            score_if_prev_odd = dp_odd - x + current_num
            
            current_score_at_i = max(score_if_prev_even, score_if_prev_odd)
            
            # Update dp_even: it's the maximum of its old value 
            # (max score ending at an earlier even number)
            # or the new score current_score_at_i (score ending at current_num).
            dp_even = max(dp_even, current_score_at_i)
            
        else:  # current_num is odd
            # Option 1: Previous element in path was odd.
            # Score = dp_odd + current_num (No penalty: odd -> odd)
            score_if_prev_odd = dp_odd + current_num

            # Option 2: Previous element in path was even.
            # Score = dp_even - x + current_num (Penalty x: even -> odd)
            score_if_prev_even = dp_even - x + current_num
            
            current_score_at_i = max(score_if_prev_odd, score_if_prev_even)
            
            dp_odd = max(dp_odd, current_score_at_i)
        
        # The overall maximum score could be for a path ending at any element.
        max_total_score = max(max_total_score, current_score_at_i)
            
    # The problem asks for an integer return type.
    return int(max_total_score)
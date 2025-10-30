from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        
        def get_max_digit(n: int) -> int:
            """Helper function to find the maximum digit of a number."""
            # A concise way is to convert the number to a string and find the max character.
            return int(max(str(n)))

        max_sum = -1
        # Dictionary to store the largest number encountered so far for each max digit.
        # Key: max digit (1-9)
        # Value: largest number seen with this max digit
        max_num_for_digit = {}

        for num in nums:
            # Find the maximum digit of the current number.
            d = get_max_digit(num)
            
            # Check if we have already seen a number with the same max digit.
            if d in max_num_for_digit:
                # If yes, we can form a pair.
                # Update the overall max_sum with this new pair's sum if it's larger.
                max_sum = max(max_sum, num + max_num_for_digit[d])
                
                # Update the stored number for this max digit if the current number is larger.
                # This ensures we always pair with the largest number seen so far in the group.
                max_num_for_digit[d] = max(max_num_for_digit[d], num)
            else:
                # If this is the first number with this max digit, store it.
                max_num_for_digit[d] = num
                
        return max_sum
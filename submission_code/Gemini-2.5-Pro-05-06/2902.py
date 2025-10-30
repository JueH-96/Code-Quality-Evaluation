import collections
from typing import List

class Solution:
    def _get_max_digit(self, n: int) -> int:
        """
        Helper function to find the maximum digit in a positive integer n.
        Constraints: Based on problem, 1 <= n <= 10^4.
        """
        # Using string conversion: efficient for small numbers of digits.
        # Example: n=571 -> str(n)="571" -> max("571")='7' -> int('7')=7
        s = str(n)
        return int(max(s))

    def maxSum(self, nums: List[int]) -> int:
        max_overall_sum = -1
        
        # Step 1: Group numbers by their maximum digit.
        # 'groups' will map a max_digit (e.g., 7) to a list of numbers 
        # from 'nums' that have this max_digit (e.g., [71, 17, 70]).
        groups = collections.defaultdict(list)
        
        for num in nums:
            # Find the maximum digit for the current number.
            max_digit_of_num = self._get_max_digit(num)
            # Add the number to the list associated with its max_digit.
            groups[max_digit_of_num].append(num)
            
        # Step 2: Iterate through each group to find the best pair sum.
        # For each max_digit that appeared in 'nums':
        for max_digit_key in groups:
            numbers_in_group = groups[max_digit_key]
            
            # A pair can only be formed if there are at least two numbers
            # with this same max_digit.
            if len(numbers_in_group) >= 2:
                # To find the maximum sum for this group, we need the two largest numbers.
                # Sorting the list in descending order makes it easy to pick them.
                # For N <= 100, sort is efficient enough.
                numbers_in_group.sort(reverse=True)
                
                # The sum of the two largest numbers in this group.
                current_pair_sum = numbers_in_group[0] + numbers_in_group[1]
                
                # Update the overall maximum sum if this group's sum is larger.
                if current_pair_sum > max_overall_sum:
                    max_overall_sum = current_pair_sum
                    
        # If no such pair was found (e.g., all numbers have unique max_digits,
        # or all groups have fewer than 2 numbers), max_overall_sum will remain -1.
        return max_overall_sum
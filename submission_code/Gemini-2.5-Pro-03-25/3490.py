import math
from typing import List

class Solution:
    """
    This class provides a solution to find the maximum length of a valid subsequence.
    A subsequence `sub` of length `x` is valid if the parity of the sum of adjacent elements 
    is constant throughout the subsequence:
    (sub[0] + sub[1]) % 2 == (sub[1] + sub[2]) % 2 == ... == (sub[x - 2] + sub[x - 1]) % 2.

    The parity of the sum (a + b) % 2 depends on the parities of a and b:
    - If a and b have the same parity (both even or both odd), (a + b) % 2 == 0.
    - If a and b have different parities (one even, one odd), (a + b) % 2 == 1.

    There are two cases for the constant parity `P`:
    Case 1: P = 0.
    The condition (sub[i] + sub[i+1]) % 2 == 0 implies that sub[i] and sub[i+1] must have the same parity for all i.
    This means all elements in the valid subsequence must have the same parity.
    The longest such subsequence consists of either all odd numbers from `nums` or all even numbers from `nums`.
    Its length is max(count_odd, count_even), where count_odd and count_even are the total counts of odd and even numbers in `nums`.

    Case 2: P = 1.
    The condition (sub[i] + sub[i+1]) % 2 == 1 implies that sub[i] and sub[i+1] must have different parities for all i.
    This means the elements in the valid subsequence must alternate in parity (e.g., odd, even, odd, ... or even, odd, even, ...).
    We can find the length of the longest alternating subsequence using a greedy approach.
    We calculate the length of the longest alternating subsequence starting with an odd number and the length of the longest one starting with an even number. The maximum of these two lengths is the maximum length for P=1.

    The final answer is the maximum length found across both cases (P=0 and P=1).
    """
    def maximumLength(self, nums: List[int]) -> int:
        """
        Finds the maximum length of a valid subsequence of nums.

        Args:
          nums: A list of integers. Length is at least 2.

        Returns:
          The maximum length of a valid subsequence.
        """
        
        # Initialize counts for odd and even numbers
        count_odd = 0
        count_even = 0
        
        # Initialize lengths for alternating subsequences
        
        # length_alt_odd_start: Tracks the length of the longest alternating subsequence 
        # found so far that starts with an odd number.
        length_alt_odd_start = 0
        # current_parity_needed_for_odd_start: Tracks the required parity for the next element 
        # to extend the alternating subsequence starting with odd. Initially needs odd (1).
        # We use integers 0 for even, 1 for odd.
        current_parity_needed_for_odd_start = 1 
        
        # length_alt_even_start: Tracks the length of the longest alternating subsequence
        # found so far that starts with an even number.
        length_alt_even_start = 0
        # current_parity_needed_for_even_start: Tracks the required parity for the next element
        # to extend the alternating subsequence starting with even. Initially needs even (0).
        current_parity_needed_for_even_start = 0 
        
        # Iterate through the input array `nums` once to compute all necessary values
        for num in nums:
            # Determine the parity of the current number (0 for even, 1 for odd)
            parity = num % 2
            
            # Update counts of odd and even numbers
            if parity == 0:
                count_even += 1
            else: # parity == 1
                count_odd += 1
                
            # Update the length of the longest alternating subsequence starting with odd
            # Check if the current number's parity matches what is needed for the subsequence
            # starting with odd.
            if parity == current_parity_needed_for_odd_start:
                # If it matches, increment the length of this type of subsequence
                length_alt_odd_start += 1
                # Flip the required parity for the next element (0 becomes 1, 1 becomes 0).
                # This ensures the alternating pattern.
                current_parity_needed_for_odd_start = 1 - current_parity_needed_for_odd_start
                
            # Update the length of the longest alternating subsequence starting with even
            # Check if the current number's parity matches what is needed for the subsequence
            # starting with even.
            if parity == current_parity_needed_for_even_start:
                # If it matches, increment the length of this type of subsequence
                length_alt_even_start += 1
                # Flip the required parity for the next element.
                current_parity_needed_for_even_start = 1 - current_parity_needed_for_even_start
                
        # After iterating through all numbers, we have the necessary information.
        
        # Calculate the maximum length for Case P=0 (all elements same parity).
        # This is the maximum of the count of odd numbers and the count of even numbers.
        max_len_P0 = max(count_odd, count_even)
        
        # Calculate the maximum length for Case P=1 (alternating parity elements).
        # This is the maximum length of an alternating subsequence, which can start either odd or even.
        max_len_P1 = max(length_alt_odd_start, length_alt_even_start)
        
        # The overall maximum length is the maximum length found across both cases (P=0 and P=1).
        return max(max_len_P0, max_len_P1)
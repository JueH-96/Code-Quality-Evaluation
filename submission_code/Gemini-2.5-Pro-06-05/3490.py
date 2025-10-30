from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        """
        Calculates the length of the longest valid subsequence.

        A valid subsequence has a constant parity of sum for adjacent elements.
        This leads to four possible patterns for the parities of the subsequence:
        1. All even: [even, even, ...]
        2. All odd: [odd, odd, ...]
        3. Alternating, starting with even: [even, odd, even, ...]
        4. Alternating, starting with odd: [odd, even, odd, ...]
        We calculate the max length for each pattern and return the overall maximum.
        """

        # Lengths for subsequences with all-even or all-odd numbers
        even_count = 0
        odd_count = 0
        
        # Length for alternating subsequence starting with even
        alt_even_start_len = 0
        next_parity_for_even = 0  # Start looking for even (parity 0)
        
        # Length for alternating subsequence starting with odd
        alt_odd_start_len = 0
        next_parity_for_odd = 1  # Start looking for odd (parity 1)

        for num in nums:
            parity = num % 2
            
            # Cases 1 & 2: Update counts for all-even/all-odd subsequences
            if parity == 0:
                even_count += 1
            else:
                odd_count += 1
            
            # Case 3: Greedily build the alternating sequence starting with even
            if parity == next_parity_for_even:
                alt_even_start_len += 1
                next_parity_for_even = 1 - parity  # Flip expectation
            
            # Case 4: Greedily build the alternating sequence starting with odd
            if parity == next_parity_for_odd:
                alt_odd_start_len += 1
                next_parity_for_odd = 1 - parity  # Flip expectation
                
        return max(even_count, odd_count, alt_even_start_len, alt_odd_start_len)
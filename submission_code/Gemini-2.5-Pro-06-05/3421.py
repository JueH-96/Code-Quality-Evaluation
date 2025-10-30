from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # We need to find pairs (i, j) with i < j such that (hours[i] + hours[j]) % 24 == 0.
        # This is equivalent to (hours[i] % 24 + hours[j] % 24) % 24 == 0.
        # We can use a frequency map (an array of size 24) to store the counts
        # of remainders encountered so far.
        
        remainder_counts = [0] * 24
        pair_count = 0
        
        for h in hours:
            # Calculate the remainder of the current hour when divided by 24.
            remainder = h % 24
            
            # To form a sum that is a multiple of 24, we need a complement.
            # If the current remainder is r, we need a previous remainder r_prev such that
            # (r + r_prev) is a multiple of 24.
            # The required complement is (24 - r) % 24.
            complement = (24 - remainder) % 24
            
            # The number of new pairs we can form is the number of times we have
            # already seen the complement remainder.
            pair_count += remainder_counts[complement]
            
            # After checking for pairs, we update the frequency map for the current remainder.
            # This ensures we only count pairs (i, j) with i < j, as we pair the
            # current element with previously seen elements.
            remainder_counts[remainder] += 1
            
        return pair_count
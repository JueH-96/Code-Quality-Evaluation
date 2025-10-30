from typing import List
from itertools import combinations

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        """
        Given a list of times in hours, returns the number of pairs i, j where i < j and hours[i] + hours[j] is a multiple of 24.
        """
        valid_pairs = 0
        # Generate all unique pairs from the list
        for pair in combinations(hours, 2):
            # If the sum of hours in the pair is a multiple of 24, increment the count of valid pairs
            if sum(pair) % 24 == 0:
                valid_pairs += 1
        return valid_pairs
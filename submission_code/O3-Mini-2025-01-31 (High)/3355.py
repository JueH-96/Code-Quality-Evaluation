from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # First, compute the total score for all levels:
        # For a level, score = +1 if possible[i] == 1, and -1 if possible[i] == 0.
        total = 0
        for p in possible:
            if p == 1:
                total += 1
            else:
                total -= 1

        # Let prefix be the cumulative score that Alice gets from playing the first m levels.
        # Note: Alice must play at least 1 level and Bob must play at least 1 level,
        # so m ranges from 1 to n-1.
        prefix = 0
        for m in range(1, n):
            # Update the cumulative prefix using the (m-1)th level.
            if possible[m - 1] == 1:
                prefix += 1
            else:
                prefix -= 1

            # Bob's score is total - prefix. 
            # We need Alice's score (prefix) to be strictly greater than Bob's:
            #    prefix > total - prefix   -->   2 * prefix > total.
            if 2 * prefix > total:
                return m

        # If no valid m is found, return -1.
        return -1
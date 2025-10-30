from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Compute total sum S of scores
        S = sum(2 * p - 1 for p in possible)
        current_sum = 0  # Alice's cumulative score
        for k in range(1, n):  # k is the number of levels Alice plays, from 1 to n-1
            current_sum += 2 * possible[k - 1] - 1  # Add score of the k-th level (index k-1)
            if 2 * current_sum > S:  # Check if Alice's score > Bob's score
                return k
        return -1  # No such k found, return -1
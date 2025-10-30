from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n < 2:
            return -1  # According to constraints, n >= 2
        
        # Compute prefix sums for Alice's possible points
        prefix = [0] * n
        prefix[0] = 1 if possible[0] else -1
        for i in range(1, n):
            prefix[i] = prefix[i-1] + (1 if possible[i] else -1)
        
        # Compute suffix sums for Bob's possible points
        suffix = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = (1 if possible[i] else -1) + suffix[i+1]
        
        # Check each possible split point k
        for k in range(1, n):
            alice_points = prefix[k-1]
            bob_points = suffix[k]
            if alice_points > bob_points:
                return k
        
        return -1
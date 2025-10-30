from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        alice_score = 0
        bob_score = 0
        min_levels = float('inf')
        
        for i in range(n):
            if possible[i] == 1:
                alice_score += 1
            else:
                alice_score -= 1
            
            # Calculate Bob's score if he plays the rest of the levels
            bob_score = sum(possible[i+1:]) - (n - i - 1 - sum(possible[i+1:]))
            
            # Check if Alice's score is greater than Bob's score
            if alice_score > bob_score and i >= 0:
                min_levels = min(min_levels, i + 1)
        
        # If Alice can't gain more points than Bob, return -1
        if min_levels == float('inf'):
            return -1
        else:
            return min_levels
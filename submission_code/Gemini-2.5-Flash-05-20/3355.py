import collections
from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)

        # Calculate the total score of all levels if played optimally.
        # A 1 in 'possible' means +1 point, a 0 means -1 point.
        total_score = 0
        for p_val in possible:
            total_score += (1 if p_val == 1 else -1)

        alice_score = 0
        # Alice plays levels from index 0 up to index k.
        # The number of levels Alice plays is k + 1.
        
        # According to the constraints: "each player must play at least 1 level".
        # This implies:
        # Alice plays at least 1 level: k + 1 >= 1  => k >= 0
        # Bob plays at least 1 level: n - (k + 1) >= 1 => n - k - 1 >= 1 => k <= n - 2
        # So, k (the last index Alice plays) ranges from 0 to n-2.
        # The loop iterates k from 0 up to n-2 (exclusive of n-1).
        for k in range(n - 1): 
            # Add the score for the current level k to Alice's score
            alice_score += (1 if possible[k] == 1 else -1)
            
            # Bob's score is the total score of all levels minus Alice's current score
            bob_score = total_score - alice_score
            
            # Check if Alice's score is strictly greater than Bob's score
            if alice_score > bob_score:
                # If the condition is met, k+1 is the minimum number of levels Alice needs to play.
                # We return k+1 immediately because we are iterating k in increasing order.
                return k + 1
        
        # If the loop finishes without Alice ever scoring more than Bob, 
        # it's not possible, so return -1.
        return -1
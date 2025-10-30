from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        """
        Calculates the minimum number of levels Alice should play to gain more points than Bob.

        Args:
          possible: A binary array where possible[i] = 1 means level i is clearable
                    and possible[i] = 0 means it's not.

        Returns:
          The minimum number of levels Alice must play to have a higher score than Bob.
          Returns -1 if it's not possible.
        """
        n = len(possible)
        
        # The score for a level is +1 if clearable (possible[i]=1) and -1 if not (possible[i]=0).
        # This can be computed as (2 * possible[i] - 1).
        # First, calculate the total score if one person played all levels.
        total_score = sum(2 * p - 1 for p in possible)
        
        alice_score = 0
        
        # Alice must play k levels, where 1 <= k <= n-1, as Bob must play at least one level.
        # We iterate through the possible split points. 'i' is the 0-based index of the last level Alice plays.
        # The number of levels Alice plays is k = i + 1.
        for i in range(n - 1):
            # Update Alice's score with the outcome of the i-th level.
            alice_score += 2 * possible[i] - 1
            
            # We need to find when Alice's score is strictly greater than Bob's score.
            # Let S_A be Alice's score and S_B be Bob's score.
            # We want S_A > S_B.
            # We know S_A + S_B = total_score, so S_B = total_score - S_A.
            # The condition becomes S_A > total_score - S_A, which simplifies to 2 * S_A > total_score.
            
            if 2 * alice_score > total_score:
                # Since we are iterating from the smallest possible k (k=1, 2, ...),
                # the first time this condition is met gives us the minimum number of levels.
                return i + 1
                
        # If the loop completes, it means for all valid splits,
        # Alice never scores more than Bob.
        return -1
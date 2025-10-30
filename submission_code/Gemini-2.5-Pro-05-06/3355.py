from typing import List

class Solution:
  def minimumLevels(self, possible: List[int]) -> int:
    n = len(possible)
    
    # Calculate TotalScore T
    # Score for a level i is 1 if possible[i]==1, else -1.
    # This is equivalent to (2 * possible[i] - 1).
    total_score = 0
    for i in range(n):
        if possible[i] == 1:
            total_score += 1
        else:
            total_score -= 1
            
    # Initialize current score for Alice
    alice_score_current = 0
    
    # Iterate k from 1 to n-1.
    # k is the number of levels Alice plays (levels 0 to k-1).
    # Bob plays levels k to n-1.
    # Both Alice (k levels) and Bob (n-k levels) must play at least 1 level.
    # The loop range(1, n) for k ensures k is in [1, n-1].
    # So Alice plays k >= 1 levels, Bob plays n-k >= 1 levels.
    for k in range(1, n): 
        # Alice completes level k-1 (0-indexed). Update her score.
        score_for_level_k_minus_1 = 0
        if possible[k-1] == 1:
            score_for_level_k_minus_1 = 1
        else:
            score_for_level_k_minus_1 = -1
        
        alice_score_current += score_for_level_k_minus_1
        
        # Bob's score would be (total_score - alice_score_current).
        # We need: alice_score_current > bob_score_current
        # This simplifies to: 2 * alice_score_current > total_score
        
        if (2 * alice_score_current) > total_score:
            # This is the first k for which the condition holds, so it's the minimum.
            return k
            
    # If loop finishes, no k satisfies the condition.
    return -1
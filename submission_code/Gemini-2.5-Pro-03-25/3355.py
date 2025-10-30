import math # Import math is not strictly needed, but it's good practice to include standard library imports if used. In this case, it is not used. We can remove it.
from typing import List # Import List for type hinting

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        """
        Calculates the minimum number of levels Alice must play such that her score is strictly greater than Bob's score.
        Alice plays the first k levels (indices 0 to k-1), and Bob plays the remaining levels (indices k to n-1).
        Both players must play at least 1 level. Thus, k must be in the range [1, n-1].
        A score of +1 is gained for clearing a level (possible[i] == 1), and -1 for failing (possible[i] == 0).

        Args:
            possible: A list of integers (0 or 1) where 1 indicates a clearable level and 0 indicates an impossible level.
                      The length of the list, n, is the total number of levels.

        Returns:
            The minimum integer k (number of levels Alice plays) such that Alice's score > Bob's score.
            Returns -1 if no such k exists in the range [1, n-1].
        
        The approach uses prefix sums and the relationship between Alice's score, Bob's score, and the total score.
        Let A_k be Alice's score after playing k levels.
        Let B_k be Bob's score playing levels k to n-1.
        Let TotalScore be the sum of scores for all levels.
        We know A_k + B_k = TotalScore.
        We want the minimum k such that A_k > B_k.
        Substituting B_k = TotalScore - A_k, the condition becomes A_k > TotalScore - A_k.
        This simplifies to 2 * A_k > TotalScore.
        We can compute TotalScore once. Then iterate k from 1 to n-1, maintaining A_k (Alice's cumulative score).
        The first k that satisfies the condition 2 * A_k > TotalScore is the minimum k.
        """
        n = len(possible)
        
        # Calculate the total score achievable across all n levels.
        # The score for level i is 1 if possible[i] == 1, and -1 if possible[i] == 0.
        # This can be compactly written as 2 * possible[i] - 1.
        # We use a generator expression with sum() for conciseness.
        total_score = sum(2 * p - 1 for p in possible)

        # Initialize Alice's running score. This will store A_k.
        current_alice_score = 0
        
        # Iterate through possible values of k, which is the number of levels Alice plays.
        # k ranges from 1 to n-1 because both Alice and Bob must play at least one level.
        # The loop range(1, n) generates integers 1, 2, ..., n-1.
        for k in range(1, n):
            # Alice plays levels 0 to k-1. In this iteration, k represents the number of levels Alice has played.
            # The last level Alice completed to reach k levels is level k-1.
            # Update Alice's score by adding the score obtained from level k-1.
            score_k_minus_1 = 2 * possible[k-1] - 1 # Calculate score for level k-1
            current_alice_score += score_k_minus_1 # Update Alice's cumulative score A_k
            
            # Check the condition derived: 2 * A_k > TotalScore
            if 2 * current_alice_score > total_score:
                # Since we are iterating k in increasing order (1, 2, ...),
                # the first k that satisfies the condition is the minimum k required.
                return k
                
        # If the loop completes without finding a suitable k, it means
        # for all possible splits (k=1 to n-1), Alice's score is not strictly greater than Bob's.
        # In this scenario, it's impossible for Alice to achieve a higher score than Bob under the constraints.
        # Return -1 as specified by the problem statement.
        return -1
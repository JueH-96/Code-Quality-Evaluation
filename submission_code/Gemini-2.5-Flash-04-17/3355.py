from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        """
        Finds the minimum number of levels Alice should play to gain more points than Bob,
        assuming both play optimally to maximize their points.

        Args:
            possible: A binary array where possible[i] == 1 if the i-th level
                      is possible to clear, and 0 otherwise.

        Returns:
            The minimum number of levels Alice should play (1-indexed).
            Returns -1 if it's not possible for Alice to score more than Bob.
        """
        n = len(possible)

        # Calculate the score for each level if played optimally:
        # If possible[i] == 1, the player clears it and gains 1 point.
        # If possible[i] == 0, the player fails it and loses 1 point.
        # The score for level i is therefore 2 * possible[i] - 1.

        # Calculate the total score of the game if all levels were played by one player.
        total_score = 0
        for p in possible:
            total_score += (2 * p - 1)

        # alice_score will accumulate the score Alice gets for the levels she plays.
        alice_score = 0

        # Alice plays the first k levels (indices 0 to k-1).
        # Bob plays the remaining n-k levels (indices k to n-1).
        # The problem states that both players must play at least 1 level.
        # This means Alice must play at least 1 level (k >= 1),
        # and Bob must play at least 1 level (n-k >= 1, which implies k <= n-1).
        # So, the number of levels Alice plays, k, must be in the range [1, n-1].
        # We are looking for the minimum such k.
        for k in range(1, n):
            # In the k-th iteration of the loop (where k is the number of levels Alice plays),
            # we add the score of the k-th level (which is at index k-1) to Alice's total.
            # After this step, alice_score holds the sum of scores for levels 0 through k-1.
            score_of_level_k_minus_1 = 2 * possible[k-1] - 1
            alice_score += score_of_level_k_minus_1

            # If Alice plays k levels, Bob plays the remaining n-k levels.
            # Bob's total score is the total game score minus Alice's score for her levels.
            bob_score = total_score - alice_score

            # We check if Alice's score is strictly greater than Bob's score.
            if alice_score > bob_score:
                # If this condition is met, the current value of k is a valid solution.
                # Since we are iterating k in increasing order starting from 1,
                # the first k that satisfies the condition is the minimum such k.
                return k

        # If the loop completes without finding any k (from 1 to n-1)
        # where Alice scores strictly more than Bob, then it's not possible.
        return -1
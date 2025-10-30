class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        """
        We use a greedy, simulation-style approach with the following key ideas:

        1) Sort all enemy energies in ascending order.
        2) Maintain two pointers:
           - left  = 0  (points to the smallest-cost unmarked enemy)
           - right = n-1 (points to the largest-cost unmarked enemy)
        3) Maintain a variable "points" = 0, which counts how many times
           we successfully perform the first operation (#1).
        4) At each step:
           - If we can afford the cost of the enemy at 'left' (i.e. currentEnergy >= E[left]),
             then we perform operation #1 as many times in a row as possible on that same enemy.
               * Let x = currentEnergy // E[left].
                 We gain x points, and reduce our energy by x * E[left].
               * We do NOT mark this enemy, so we do NOT move 'left'.
                 (Because we may be able to do #1 again on this same unmarked enemy after
                  we gain energy from marking some other enemy via operation #2.)
           - Otherwise, if we cannot afford E[left] but we already have at least 1 point,
             we perform operation #2 on the largest unmarked enemy at 'right'. This
             increases our currentEnergy by E[right], and we mark that enemy (so right -= 1).
           - If neither of these can happen (we cannot afford E[left], and points == 0),
             we break out, as no more operations are possible.
        5) The variable 'points' at the end of this loop is the maximum possible.

        This method correctly reproduces the provided examples and runs efficiently:
        - Sorting is O(n log n).
        - The main loop, due to "batching" multiple #1 operations into a single step,
          runs in O(n) after sorting.
        """

        import sys
        # In case of large I/O (not strictly necessary here, but good practice sometimes)
        data = enemyEnergies

        data.sort()
        points = 0
        left = 0
        right = len(data) - 1

        while left <= right:
            # If we can afford the smallest unmarked enemy, do as many #1 as possible on it.
            if currentEnergy >= data[left]:
                # Number of times we can pay for data[left]
                x = currentEnergy // data[left]

                # Add x to our total "points"
                points += x

                # Decrease our energy by x * data[left]
                currentEnergy -= x * data[left]

            else:
                # Can't afford data[left]. If we have >= 1 point, we can do #2 on the largest unmarked enemy.
                if points >= 1:
                    currentEnergy += data[right]
                    right -= 1
                else:
                    # We have 0 points and can't afford E[left], so we're stuck
                    break

        return points
from typing import List
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        # Since n is small (n <= 8), we can try every permutation of locks.
        # The sword starts with factor X = 1 and energy = 0.
        # For a given lock with required energy s and current factor X, the minimum minutes needed
        # is the smallest m such that m * X >= s, which is m = ceil(s / X).
        # After breaking a lock the energy resets to 0 and factor is increased by K.
        # Thus if we break a lock in the j-th step, the factor will be:
        #    current_factor = 1 + j*K   (with j = 0, 1, 2, …)
        # For a given order of locks (permutation perm), the total time is:
        #   total_time = sum_{j=0}^{n-1} ceil(perm[j] / (1 + j*K))
        # We simply try all orders and return the minimum total time.
        
        best_time = float('inf')
        n = len(strength)
        
        for perm in itertools.permutations(strength):
            time_taken = 0
            for i, s in enumerate(perm):
                current_factor = 1 + i * K
                # Using integer arithmetic to simulate math.ceil(s / current_factor)
                time_taken += (s + current_factor - 1) // current_factor
            best_time = min(best_time, time_taken)
        
        return best_time

# Example usage and testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.findMinimumTime([3, 4, 1], 1))  # Expected output: 4
    # Example 2:
    print(sol.findMinimumTime([2, 5, 4], 2))  # Expected output: 5
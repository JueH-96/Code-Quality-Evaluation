from itertools import permutations
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        best_time = float('inf')
        
        # Try all orders in which Bob can break the locks.
        for order in permutations(strength):
            minutes = 0
            factor = 1  # initial factor X
            for req in order:
                # Determine how many minutes are needed for current lock
                # In every minute, the sword's energy increases by factor and energy resets after break,
                # so the minimum minutes needed for the current lock is ceil(req / factor).
                needed = (req + factor - 1) // factor  # equivalent to ceil(req/factor)
                minutes += needed
                # Increase the factor after breaking the lock
                factor += K
                # Early prune: if already exceeding our best solution, skip further calculation.
                if minutes >= best_time:
                    break
            best_time = min(best_time, minutes)
        
        return best_time
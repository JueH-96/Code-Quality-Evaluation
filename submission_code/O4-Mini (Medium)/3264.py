from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort energies ascending for the cheapest fight,
        # and descending for the best refuel.
        if not enemyEnergies:
            return 0
        n = len(enemyEnergies)
        asc = sorted(enemyEnergies)
        desc = sorted(enemyEnergies, reverse=True)
        
        points = 0
        refuels_used = 0  # how many enemies we've used for op2 (marking)
        
        # We can only fight (op1) if:
        #   - we have at least one unmarked enemy (refuels_used < n)
        #   - our currentEnergy >= smallest enemy energy asc[0]
        # We can only refuel (op2) if:
        #   - we have at least 1 point
        #   - there is at least one unmarked enemy left (refuels_used < n)
        #
        # We loop, always:
        #   1) fight as many times as we can on the cheapest enemy
        #   2) if we can't fight but have >=1 point and a refuel left, refuel on
        #      the largest remaining energy enemy
        #   3) otherwise stop
        while True:
            # 1) Fight cheaply as long as possible
            if refuels_used < n and currentEnergy >= asc[0]:
                # fight in batch: number of times we can subtract asc[0]
                times = currentEnergy // asc[0]
                points += times
                currentEnergy -= times * asc[0]
            # 2) Can't fight any more: try to refuel if we have points and refuels left
            elif points >= 1 and refuels_used < n:
                # refuel on the next largest unmarked enemy
                currentEnergy += desc[refuels_used]
                refuels_used += 1
                # refuel does not change points
            else:
                # neither fight nor refuel is possible
                break
        
        return points
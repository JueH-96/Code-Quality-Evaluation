import math
from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort enemy energies in ascending order.
        # This makes enemyEnergies[0] the cheapest enemy to attack
        # and enemyEnergies[len-1] the most expensive (best for healing).
        enemyEnergies.sort()
        
        n = len(enemyEnergies)
        
        # If we cannot even afford to attack the cheapest enemy (enemyEnergies[0]) initially,
        # and we have 0 points, we can't perform any operation to gain points.
        # Constraints: 1 <= enemyEnergies.length, so enemyEnergies[0] is always valid.
        if currentEnergy < enemyEnergies[0]:
            return 0
        
        # Initialize points and two pointers
        points = 0
        # l points to the cheapest enemy. For attacks, we always consider enemyEnergies[0]
        # as attacks don't mark enemies. So 'l' will effectively remain 0 for attack cost.
        l = 0 
        # r points to the most expensive available enemy, used for healing.
        r = n - 1 
        
        # The main loop continues as long as there are unmarked enemies available (l <= r).
        while l <= r:
            # Phase 1: Attack (Gain points)
            # We always target the cheapest enemy (enemyEnergies[l], which is effectively enemyEnergies[0]).
            # Gain as many points as possible with the current energy by repeatedly attacking the cheapest enemy.
            if currentEnergy >= enemyEnergies[l]:
                # Calculate how many times we can attack enemyEnergies[l]
                num_attacks = currentEnergy // enemyEnergies[l]
                points += num_attacks
                currentEnergy %= enemyEnergies[l]
            
            # Phase 2: Recharge (if necessary and possible)
            # If we don't have enough energy to attack the cheapest enemy (enemyEnergies[l]),
            # we need to consider increasing currentEnergy using Operation 2.
            else: # currentEnergy < enemyEnergies[l]
                if points > 0 and l < r:
                    # Use the most expensive available unmarked enemy to regain energy.
                    # This enemy gets marked, so we move the 'r' pointer inwards.
                    currentEnergy += enemyEnergies[r]
                    r -= 1
                else:
                    # We cannot attack (currentEnergy is too low), AND
                    # we either have no points to sacrifice (points == 0) OR
                    # we have no more expensive enemies to sacrifice (l == r, meaning only the cheapest enemy is left).
                    # In this scenario, we cannot gain any more points.
                    break
                    
        return points
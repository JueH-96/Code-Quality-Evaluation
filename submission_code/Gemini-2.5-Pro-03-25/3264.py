import math
from typing import List # Ensure List is imported for type hinting

class Solution:
    """
    Solves the problem of finding the maximum points achievable by performing operations on enemies.
    The operations involve spending energy to gain points or marking enemies to gain energy.
    """
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        """
        Calculates the maximum points that can be obtained by optimally applying the two operations.

        Args:
          enemyEnergies: A list of integers representing energy values of enemies. 
                         Constraints: 1 <= len(enemyEnergies) <= 10^5, 1 <= enemyEnergies[i] <= 10^9.
          currentEnergy: An integer representing the initial energy.
                         Constraints: 0 <= currentEnergy <= 10^9.

        Returns:
          An integer denoting the maximum points achievable. The result can be large, potentially exceeding
          standard 64-bit integer limits, but Python's arbitrary precision integers handle this.
        """
        
        n = len(enemyEnergies)
        # Per constraints, n >= 1, so the list is never empty. No need to handle the n=0 case explicitly.
        
        # Sort the enemy energies in non-decreasing order. This allows us to easily identify
        # the enemy with the minimum energy, which is crucial for the optimal strategy.
        # Sorting takes O(N log N) time complexity.
        enemyEnergies.sort()
        
        # The enemy with the minimum energy cost for Operation 1 (gain point) is the first element after sorting.
        min_energy = enemyEnergies[0]
        
        # To gain the first point, we must use Operation 1. This requires currentEnergy >= enemyEnergies[i]
        # for some enemy i. The easiest enemy to use Operation 1 on is the one with `min_energy`.
        # If the initial `currentEnergy` is less than `min_energy`, we cannot even gain the first point.
        # In this case, the maximum points achievable is 0.
        if currentEnergy < min_energy:
            return 0
            
        # The optimal strategy identified is as follows:
        # 1. Always use Operation 1 (gain point) on the enemy with the minimum energy cost (`min_energy`),
        #    as this is the most energy-efficient way to gain points.
        # 2. To replenish energy required for Operation 1, use Operation 2 (gain energy). Operation 2
        #    marks an enemy permanently. To maximize the energy gain per marked enemy (and thus maximize
        #    the potential for future points), it's optimal to use Operation 2 on unmarked enemies
        #    with the highest energy values.
        # 3. This strategy effectively means we can conceptualize the process as converting all enemies,
        #    except the one with `min_energy`, into energy using Operation 2. This gained energy,
        #    combined with the initial `currentEnergy`, forms a total energy pool.
        # 4. This total energy pool is then spent entirely on gaining points by repeatedly applying
        #    Operation 1 on the enemy with `min_energy`.

        # Calculate the sum of energies of all enemies. This can be done efficiently using the built-in sum function.
        # Summing takes O(N) time complexity.
        total_sum_energies = sum(enemyEnergies) 
            
        # The total energy gained from using Operation 2 on all enemies except the minimum one.
        # This is equivalent to the total sum of energies minus the minimum energy.
        # We subtract `min_energy` because the enemy with minimum energy is reserved for Operation 1.
        energy_from_others = total_sum_energies - min_energy
            
        # Calculate the total effective energy pool available for spending on gaining points.
        # This consists of the initial energy plus all the energy gained from other enemies.
        total_available_energy = currentEnergy + energy_from_others
        
        # Each point costs `min_energy` when using Operation 1 on the minimum energy enemy.
        # The maximum number of points is the total available energy divided by the cost per point.
        # Integer division `//` correctly computes the floor of the division, representing the maximum
        # number of whole points that can be acquired.
        max_points = total_available_energy // min_energy
        
        # Return the calculated maximum points.
        return max_points
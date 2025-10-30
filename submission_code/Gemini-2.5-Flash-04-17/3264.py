from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        
        # Find the minimum energy among all enemies.
        # Constraints guarantee enemyEnergies is not empty and energies are >= 1.
        min_energy = min(enemyEnergies)
        
        # If initial currentEnergy is less than the minimum enemy energy,
        # we cannot perform the first operation 1 to gain any points (which requires
        # currentEnergy >= enemyEnergy).
        if currentEnergy < min_energy:
            return 0
            
        # Calculate the sum of all enemy energies.
        # Python's default integers handle large values, so overflow is not a concern
        # for sums up to 10^14, which fits comfortably.
        total_sum_energies = sum(enemyEnergies)
        
        # Optimal strategy:
        # We gain points by performing Operation 1. The cost per point is minimized by using
        # the enemy with the minimum energy (min_energy). To maximize points, we want to perform
        # Operation 1 on a min_energy enemy as many times as possible.
        
        # Operation 1 does not mark the enemy, allowing repeated use of an unmarked enemy.
        # Operation 2 allows gaining energy from an unmarked enemy, but it marks the enemy,
        # preventing further use for either operation. It requires having at least 1 point.
        
        # To maximize the total energy available to spend on Operation 1 (costing min_energy each),
        # we should leverage Operation 2 to gain energy from other enemies. The enemies
        # with the highest energies are the best candidates for Operation 2 to maximize gain.
        # However, to maximize the *total* energy gain across the entire process using Op2,
        # we should use all enemies *except* those we need to keep unmarked for Op1.
        # Since Operation 1 is most efficient with min_energy, we designate one instance
        # of a min_energy enemy to be used repeatedly for Operation 1, ensuring it is never
        # used for Operation 2 (which would mark it). All *other* enemies (including duplicate
        # min_energy enemies and all enemies with higher energy) can be used for Operation 2
        # to boost currentEnergy, as long as we have at least 1 point.
        
        # The total energy pool available to be converted into points (each costing min_energy) is:
        # Initial currentEnergy + Total energy gained from Operation 2 on all other enemies.
        # The total energy gained from Operation 2 on all enemies *except* one min_energy instance is:
        # sum(all enemy energies) - min_energy.
        
        # Total spendable energy on min_energy operations = currentEnergy + (total_sum_energies - min_energy).
        # This strategy is feasible because currentEnergy >= min_energy allows getting the first point,
        # which in turn enables Operation 2 on other enemies to increase the energy pool.
        
        # The maximum number of times we can perform Operation 1 (total points) is the total
        # spendable energy divided by the cost per operation (min_energy).
        # Integer division (//) provides the floor, correctly counting full operations.
        # Since min_energy >= 1, division by zero is not a concern.
        
        maximum_points = (currentEnergy + total_sum_energies - min_energy) // min_energy
        
        return int(maximum_points) # Ensure integer return type
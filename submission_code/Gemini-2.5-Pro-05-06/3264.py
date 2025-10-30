from typing import List

class Solution:
  def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
    # Sort enemyEnergies to easily find the minimum cost enemy.
    enemyEnergies.sort()
    
    # The enemy with the minimum energy will be used for Operation 1 (gaining points).
    # This is because Operation 1 leaves the enemy unmarked, so we can repeatedly use it.
    # To maximize points for a given amount of energy, we should use the cheapest enemy.
    min_cost_enemy_energy = enemyEnergies[0]
    
    # If currentEnergy is less than the energy of the weakest enemy,
    # we cannot gain the first point. 
    # Operation 1: requires currentEnergy >= enemyEnergies[i].
    # Operation 2: requires at least 1 point.
    # If we can't make the first move (Op1), we get 0 points and can't unlock Op2.
    # So, 0 points can be gained in total.
    if currentEnergy < min_cost_enemy_energy:
      return 0
      
    # If currentEnergy >= min_cost_enemy_energy, we can gain points.
    # The strategy, as derived in thought process, simplifies to:
    # Total points = (currentEnergy_initial + sum_other_energies) / min_cost_enemy_energy
    # where sum_other_energies is the sum of energies from all enemies EXCEPT the one
    # with min_cost_enemy_energy.
    
    # Calculate sum_other_energies.
    # This is the sum of energies of all enemies from enemyEnergies[1] onwards.
    # If enemyEnergies has only one element, this sum will be 0.
    sum_other_energies = 0
    # The loop range(1, len(enemyEnergies)) correctly handles len=1 
    # (loop doesn't run, sum_other_energies remains 0).
    for i in range(1, len(enemyEnergies)):
      sum_other_energies += enemyEnergies[i]
      
    # This is the total energy pool available to spend on the min_cost_enemy.
    total_effective_energy = currentEnergy + sum_other_energies
    
    # Calculate points using integer division, as points are discrete.
    points = total_effective_energy // min_cost_enemy_energy
    
    return points
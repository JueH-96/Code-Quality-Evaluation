from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        """
        Calculates the maximum points achievable based on enemy energies and initial energy.

        The core strategy is greedy and relies on these insights:
        1.  To gain points, we should always defeat the enemy with the minimum energy cost. This maximizes
            the number of points we get for a given amount of energy.
        2.  To gain energy, we should absorb the enemy with the maximum energy. This maximizes our
            energy gain, allowing us to defeat the weakest enemy more times.
        3.  The "absorb" operation requires having at least 1 point. The "defeat" operation requires
            having enough energy. This creates a cycle: use energy to get a point, then use the point
            to get more energy.

        The optimal sequence of operations can be simplified:
        - First, sort the `enemyEnergies` array to easily identify the weakest and strongest enemies.
        - The weakest enemy is `enemyEnergies[0]`. All other enemies are stronger.
        - To start the point-gaining cycle, we must be able to defeat at least one enemy. If our
          `currentEnergy` is less than the weakest enemy's energy (`enemyEnergies[0]`), we can't
          score any points. In this case, the answer is 0.
        - If we *can* defeat the weakest enemy, we can gain at least 1 point. With this one point,
          we unlock the ability to absorb any other unmarked enemy.
        - To maximize our final score, we should absorb all enemies except the weakest one. This
          adds their energy to our total energy pool.
        - The total energy we can accumulate is our `currentEnergy` plus the sum of energies of all
          enemies from index 1 to n-1.
        - Finally, we spend this entire accumulated energy pool on repeatedly defeating the weakest
          enemy (`enemyEnergies[0]`), as this is the most efficient way to convert energy into points.

        The maximum number of points is therefore `(total_accumulated_energy // weakest_enemy_energy)`.
        """
        
        enemyEnergies.sort()
        
        # If the initial energy is less than the energy of the weakest enemy,
        # it's impossible to score any points.
        if currentEnergy < enemyEnergies[0]:
            return 0
        
        # Calculate the total energy pool we can use. This consists of our initial energy
        # plus the energy of all enemies we choose to absorb. To maximize points, we should
        -        # absorb all enemies except for the weakest one.
        # Note: sum over an empty slice enemyEnergies[1:] is 0, correctly handling the n=1 case.
        total_energy_pool = currentEnergy + sum(enemyEnergies[1:])
        
        # The maximum points are obtained by spending the entire energy pool on defeating
        # the weakest enemy as many times as possible.
        weakest_enemy_energy = enemyEnergies[0]
        max_points = total_energy_pool // weakest_enemy_energy
        
        return max_points
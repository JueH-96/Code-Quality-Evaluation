from typing import List
from functools import cmp_to_key

class Solution:
  """
  Solves the problem of finding the minimum total damage Bob receives from enemies using a greedy strategy.
  The greedy choice is based on prioritizing enemies with a smaller ratio of 'time to kill' / 'damage per second'.
  This strategy is proven optimal using an exchange argument.
  """
  def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
    """
    Calculates the minimum total damage Bob receives until all enemies are defeated.

    Args:
      power: Bob's attack power per second. Constraint: 1 <= power <= 10^4.
      damage: A list where damage[i] is the damage per second dealt by enemy i. Constraint: 1 <= damage[i] <= 10^4.
      health: A list where health[i] is the health points of enemy i. Constraint: 1 <= health[i] <= 10^4.
      Both lists have the same length n, where 1 <= n <= 10^5.

    Returns:
      The minimum total damage Bob receives. The result can be large, so Python's arbitrary precision integers are used.
    """
    n = len(damage)
    # Create a list of tuples, each representing an enemy with its characteristics relevant to the strategy:
    # (time_to_kill, damage_per_second)
    enemies = []
    for i in range(n):
        # Calculate time needed to defeat enemy i using integer arithmetic for ceiling division.
        # t_k = ceil(health[i] / power) 
        # The formula (health[i] + power - 1) // power computes ceiling division correctly for positive integers.
        # Since health[i] >= 1 and power >= 1, t_k is guaranteed to be >= 1.
        t_k = (health[i] + power - 1) // power
        d_k = damage[i]
        # Append the tuple (t_k, d_k) to the list.
        enemies.append((t_k, d_k))

    # Define the comparison function for sorting enemies based on the t_k / d_k ratio.
    # The optimal strategy is to attack enemies with smaller t_k / d_k ratio first.
    # To avoid floating point issues and potential division by zero (though constraints say d_k >= 1),
    # we use cross-multiplication: compare t_i * d_j vs t_j * d_i.
    # This comparison is equivalent to comparing t_i / d_i vs t_j / d_j for positive d_i, d_j.
    def compare_enemies(enemy_i, enemy_j):
        """
        Comparison function for two enemies based on the greedy criterion.
        Returns a negative value if enemy_i should come before enemy_j,
        a positive value if enemy_i should come after enemy_j,
        and zero if their relative order does not matter based on the criterion.
        """
        t_i, d_i = enemy_i
        t_j, d_j = enemy_j
        # Calculate the difference t_i * d_j - t_j * d_i.
        # The sign of the difference determines the relative order.
        # If t_i * d_j < t_j * d_i, then t_i/d_i < t_j/d_j, so enemy_i comes first. The function returns negative.
        # If t_i * d_j > t_j * d_i, then t_i/d_i > t_j/d_j, so enemy_j comes first. The function returns positive.
        # If equal, the function returns zero.
        diff = t_i * d_j - t_j * d_i
        return diff

    # Sort the enemies list using the custom comparison function.
    # functools.cmp_to_key converts an old-style comparison function (like compare_enemies) 
    # into a key function suitable for `sort` or `sorted`.
    enemies.sort(key=cmp_to_key(compare_enemies))

    # Initialize total damage accumulated. Python integers handle arbitrary size, preventing overflow.
    total_damage = 0
    # Calculate the initial total damage per second rate, which is the sum of damage from all enemies.
    current_total_damage_per_second = sum(damage) 

    # Iterate through the sorted enemies list, simulating the optimal attack sequence.
    for t_k, d_k in enemies:
        # Add the damage received during the t_k seconds spent attacking the current enemy.
        # The damage rate during this phase is the sum of DPS from all enemies still alive at the start of this phase.
        total_damage += t_k * current_total_damage_per_second
        
        # After t_k seconds, the current enemy is defeated.
        # Update the total damage per second rate by subtracting the defeated enemy's contribution.
        # This updated rate will be used for calculating damage in the next phase.
        current_total_damage_per_second -= d_k
            
    # Return the final calculated minimum total damage.
    return total_damage